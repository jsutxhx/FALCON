"""多跳检索器模块"""
from typing import List, Tuple, Dict
from src.data_processing.data_structures import Paper, Entity
from src.knowledge_graph.graph_storage import KnowledgeGraph
from src.retrieval.entity_similarity import EntitySimilarityCalculator
from src.retrieval.path_similarity import PathSimilarityCalculator


class MultiHopRetriever:
    """多跳检索器
    
    功能：
    - 整合实体相似度和路径相似度进行多跳检索
    - 从知识图谱中检索候选论文
    - 计算综合得分并返回top-k候选
    
    Attributes:
        entity_calculator: 实体相似度计算器
        path_calculator: 路径相似度计算器
        graph: 知识图谱对象
        entity_path_balance: 实体相似度和路径相似度的平衡权重 γ
    """
    
    def __init__(
        self,
        entity_calculator: EntitySimilarityCalculator,
        path_calculator: PathSimilarityCalculator,
        graph: KnowledgeGraph,
        entity_path_balance: float = 0.6
    ):
        """初始化多跳检索器
        
        Args:
            entity_calculator: 实体相似度计算器
            path_calculator: 路径相似度计算器
            graph: 知识图谱对象
            entity_path_balance: 实体相似度和路径相似度的平衡权重 γ（0-1之间）
                                例如0.6表示60%权重给实体相似度，40%给路径相似度
        """
        self.entity_calculator = entity_calculator
        self.path_calculator = path_calculator
        self.graph = graph
        self.entity_path_balance = entity_path_balance
    
    def _get_paper_entities(self, paper: Paper) -> Dict[str, List[Entity]]:
        """从知识图谱中获取论文的所有实体
        
        功能：
        - 通过contains边从知识图谱中获取论文的所有实体
        - 按实体类型分组
        
        Args:
            paper: 论文对象
            
        Returns:
            实体字典，键为实体类型，值为该类型的实体列表
        """
        entities_by_type: Dict[str, List[Entity]] = {}
        
        # 检查论文节点是否存在
        if paper.id not in self.graph.graph:
            return entities_by_type
        
        # 获取所有出边（contains边）
        for neighbor_id in self.graph.graph[paper.id]:
            edge_attrs = self.graph.graph[paper.id][neighbor_id]
            relation_type = edge_attrs.get("relation_type")
            
            # 只处理contains边
            if relation_type == "contains":
                # 获取实体节点
                if neighbor_id in self.graph.graph:
                    entity_node = self.graph.get_node(neighbor_id)
                    entity_type = entity_node.get("entity_type")
                    
                    if entity_type:
                        # 创建Entity对象
                        entity = Entity(
                            id=entity_node.get("id", neighbor_id),
                            text=entity_node.get("text", ""),
                            entity_type=entity_type,
                            canonical=entity_node.get("canonical", ""),
                            weight=entity_node.get("weight", 1.0),
                            embedding=None  # 如果需要embedding，需要从graph中获取
                        )
                        
                        # 按类型分组
                        if entity_type not in entities_by_type:
                            entities_by_type[entity_type] = []
                        entities_by_type[entity_type].append(entity)
        
        return entities_by_type
    
    def retrieve(self, query_paper: Paper, top_k: int = 10) -> List[Tuple[Paper, float]]:
        """检索候选论文
        
        功能：
        - 计算查询论文与所有候选论文的相似度得分
        - 综合实体相似度和路径相似度
        公式: score = γ · S_entity + (1-γ) · S_path
        其中：
        - γ 是 entity_path_balance
        - S_entity 是实体相似度
        - S_path 是路径相似度
        
        Args:
            query_paper: 查询论文对象
            top_k: 返回top-k个候选论文，默认为10
            
        Returns:
            候选论文列表，每个元素是(Paper, score)元组，按得分降序排列
        """
        # 获取所有论文节点作为候选
        paper_node_ids = self.graph.get_paper_nodes()
        
        # 排除查询论文本身
        candidate_paper_ids = [pid for pid in paper_node_ids if pid != query_paper.id]
        
        # 如果候选为空，返回空列表
        if not candidate_paper_ids:
            return []
        
        # 获取查询论文的实体
        query_entities = self._get_paper_entities(query_paper)
        
        # 计算每个候选论文的得分
        candidates_with_scores: List[Tuple[Paper, float]] = []
        
        for candidate_id in candidate_paper_ids:
            # 获取候选论文节点
            candidate_node = self.graph.get_node(candidate_id)
            
            # 创建Paper对象
            candidate_paper = Paper(
                id=candidate_node.get("id", candidate_id),
                title=candidate_node.get("title", ""),
                abstract=candidate_node.get("abstract", ""),
                authors=candidate_node.get("authors", []),
                year=candidate_node.get("year", 0),
                venue=candidate_node.get("venue", ""),
                citation_count=candidate_node.get("citation_count", 0),
                doi=candidate_node.get("doi")
            )
            
            # 获取候选论文的实体
            candidate_entities = self._get_paper_entities(candidate_paper)
            
            # 计算实体相似度
            entity_sim = self.entity_calculator.compute(query_entities, candidate_entities)
            
            # 计算路径相似度
            path_sim = self.path_calculator.compute(query_paper, candidate_paper, self.graph)
            
            # 综合得分：score = γ · S_entity + (1-γ) · S_path
            score = (
                self.entity_path_balance * entity_sim +
                (1 - self.entity_path_balance) * path_sim
            )
            
            candidates_with_scores.append((candidate_paper, score))
        
        # 按得分降序排序
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k
        return candidates_with_scores[:top_k]
