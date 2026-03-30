"""TransE知识图谱嵌入模型"""
import torch
import torch.nn as nn
import torch.nn.init as init


class TransE(nn.Module):
    """TransE知识图谱嵌入模型
    
    TransE是一个简单的知识图谱嵌入模型，通过将实体和关系映射到低维向量空间，
    使得对于三元组 (h, r, t)，有 h + r ≈ t。
    
    Attributes:
        num_entities: 实体数量
        num_relations: 关系数量
        embedding_dim: 嵌入维度，默认为128
        margin: margin ranking loss的margin值，默认为1.0
        entity_embeddings: 实体嵌入层
        relation_embeddings: 关系嵌入层
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 1.0
    ):
        """初始化TransE模型
        
        Args:
            num_entities: 实体数量
            num_relations: 关系数量
            embedding_dim: 嵌入维度，默认为128
            margin: margin ranking loss的margin值，默认为1.0
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 定义实体和关系嵌入层
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化嵌入权重
        
        使用Xavier均匀初始化来初始化实体和关系的嵌入向量。
        """
        # 初始化实体嵌入
        init.xavier_uniform_(self.entity_embeddings.weight.data)
        
        # 初始化关系嵌入
        init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # 对关系嵌入进行归一化（TransE的常见做法）
        # 将关系嵌入的L2范数归一化到1
        with torch.no_grad():
            relation_norms = torch.norm(self.relation_embeddings.weight.data, p=2, dim=1, keepdim=True)
            self.relation_embeddings.weight.data = self.relation_embeddings.weight.data / relation_norms
    
    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """计算TransE得分
        
        对于三元组 (h, r, t)，计算得分：||h + r - t||_2
        得分越小，表示三元组越合理。
        
        Args:
            head: 头实体索引，形状为 (batch_size,) 的LongTensor
            relation: 关系索引，形状为 (batch_size,) 的LongTensor
            tail: 尾实体索引，形状为 (batch_size,) 的LongTensor
            
        Returns:
            得分，形状为 (batch_size,) 的Tensor
            每个元素是 ||h + r - t||_2 的值
        """
        # 获取实体和关系的嵌入向量
        # h, r, t 的形状都是 (batch_size, embedding_dim)
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        
        # 计算 h + r - t
        # diff 的形状是 (batch_size, embedding_dim)
        diff = h + r - t
        
        # 计算L2范数 ||h + r - t||_2
        # score 的形状是 (batch_size,)
        score = torch.norm(diff, p=2, dim=-1)
        
        return score
    
    def loss(
        self,
        pos_triplets: tuple,
        neg_triplets: tuple
    ) -> torch.Tensor:
        """计算Margin Ranking Loss
        
        TransE使用margin ranking loss来训练模型，使得正样本的得分小于负样本的得分。
        损失函数为：max(0, margin + pos_score - neg_score)
        
        Args:
            pos_triplets: 正样本三元组，格式为 (head, relation, tail)
                         每个都是形状为 (batch_size,) 的LongTensor
            neg_triplets: 负样本三元组，格式为 (head, relation, tail)
                         每个都是形状为 (batch_size,) 的LongTensor
        
        Returns:
            标量损失值（平均损失）
        """
        # 计算正样本得分
        pos_head, pos_relation, pos_tail = pos_triplets
        pos_score = self.forward(pos_head, pos_relation, pos_tail)
        
        # 计算负样本得分
        neg_head, neg_relation, neg_tail = neg_triplets
        neg_score = self.forward(neg_head, neg_relation, neg_tail)
        
        # 计算margin ranking loss: max(0, margin + pos_score - neg_score)
        # 使用ReLU实现max(0, x)
        loss = torch.relu(self.margin + pos_score - neg_score)
        
        # 返回平均损失
        return loss.mean()

