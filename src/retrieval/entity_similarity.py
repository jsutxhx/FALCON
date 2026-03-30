"""类型感知实体相似度计算模块"""
from typing import List, Dict
import numpy as np
from src.data_processing.data_structures import Entity
from src.retrieval.similarity_utils import cosine_similarity


def sigmoid(x: float) -> float:
    """Sigmoid函数
    
    Args:
        x: 输入值
        
    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


class EntitySimilarityCalculator:
    """类型感知实体相似度计算器
    
    功能：
    - 计算查询实体和候选实体之间的相似度
    - 支持按实体类型分组计算
    - 使用加权平均计算类内相似度
    
    Attributes:
        type_weights: 实体类型权重字典，例如 {"task": 0.25, "method": 0.35, ...}
    """
    
    def __init__(self, type_weights: Dict[str, float]):
        """初始化实体相似度计算器
        
        Args:
            type_weights: 实体类型权重字典，键为实体类型，值为权重
        """
        self.type_weights = type_weights
    
    def _compute_intra_type_similarity(
        self,
        q_entities: List[Entity],
        c_entities: List[Entity]
    ) -> float:
        """计算同类型实体之间的相似度
        
        功能：
        - 计算查询实体和候选实体之间的加权平均相似度
        - 公式: Σ_i Σ_j w_{e_i} · w_{e_j} · cos(v_{e_i}, v_{e_j}) / 
                (Σ_i w_{e_i} · Σ_j w_{e_j})
        - 使用实体权重进行加权
        
        Args:
            q_entities: 查询实体列表（同类型）
            c_entities: 候选实体列表（同类型）
            
        Returns:
            相似度值，范围在 [-1, 1] 之间（理论上，实际通常在 [0, 1]）
            如果任一列表为空或所有实体都没有embedding，返回0.0
        """
        if not q_entities or not c_entities:
            return 0.0
        
        # 过滤出有embedding的实体
        q_entities_with_emb = [e for e in q_entities if e.embedding is not None]
        c_entities_with_emb = [e for e in c_entities if e.embedding is not None]
        
        if not q_entities_with_emb or not c_entities_with_emb:
            return 0.0
        
        # 计算分子：加权余弦相似度之和
        numerator = 0.0
        for e_i in q_entities_with_emb:
            for e_j in c_entities_with_emb:
                # 计算余弦相似度
                cos_sim = cosine_similarity(e_i.embedding, e_j.embedding)
                # 加权求和
                numerator += e_i.weight * e_j.weight * cos_sim
        
        # 计算分母：权重和的乘积
        q_weight_sum = sum(e.weight for e in q_entities_with_emb)
        c_weight_sum = sum(e.weight for e in c_entities_with_emb)
        denominator = q_weight_sum * c_weight_sum
        
        # 处理分母为0的情况
        if denominator == 0.0:
            return 0.0
        
        # 计算加权平均相似度
        similarity = numerator / denominator
        
        return float(similarity)
    
    def compute(
        self,
        query_entities: Dict[str, List[Entity]],
        candidate_entities: Dict[str, List[Entity]]
    ) -> float:
        """计算综合实体相似度
        
        功能：
        - 按实体类型分组计算相似度
        - 使用配置的类型权重加权求和
        - 应用 sigmoid 归一化到 [0, 1] 范围
        
        公式：
        S_total = σ(Σ_t w_t · S_t)
        其中：
        - w_t 是类型 t 的权重
        - S_t 是类型 t 的类内相似度（由 _compute_intra_type_similarity 计算）
        - σ 是 sigmoid 函数
        
        Args:
            query_entities: 查询实体字典，键为实体类型，值为该类型的实体列表
            candidate_entities: 候选实体字典，键为实体类型，值为该类型的实体列表
            
        Returns:
            综合相似度值，范围在 [0, 1] 之间
        """
        total_sim = 0.0
        
        # 遍历所有实体类型
        for entity_type, type_weight in self.type_weights.items():
            # 获取该类型的查询实体和候选实体
            q_entities = query_entities.get(entity_type, [])
            c_entities = candidate_entities.get(entity_type, [])
            
            # 如果两种实体都存在，计算类内相似度
            if q_entities and c_entities:
                type_sim = self._compute_intra_type_similarity(q_entities, c_entities)
                # 使用类型权重加权求和
                total_sim += type_weight * type_sim
        
        # 应用 sigmoid 归一化到 [0, 1] 范围
        normalized_sim = sigmoid(total_sim)
        
        return float(normalized_sim)

