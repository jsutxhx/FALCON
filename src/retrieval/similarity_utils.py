"""相似度计算工具模块"""
import numpy as np
from typing import Union


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量的余弦相似度
    
    功能：
    - 计算 v1 和 v2 的余弦相似度
    - 公式: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
    - 处理零向量情况（返回0.0）
    
    Args:
        v1: 第一个向量，形状为 (n,) 的numpy数组
        v2: 第二个向量，形状为 (n,) 的numpy数组，维度应与v1相同
        
    Returns:
        余弦相似度值，范围在 [-1, 1] 之间
        如果任一向量为零向量，返回0.0
    """
    # 确保输入是numpy数组
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    # 验证维度
    if v1.shape != v2.shape:
        raise ValueError(f"向量维度不匹配: v1.shape={v1.shape}, v2.shape={v2.shape}")
    
    # 计算点积
    dot_product = np.dot(v1, v2)
    
    # 计算L2范数
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # 处理零向量情况
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    # 处理数值误差（确保结果在[-1, 1]范围内）
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return float(similarity)


def batch_cosine_similarity(
    queries: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray:
    """批量计算余弦相似度
    
    功能：
    - 计算每个查询向量与所有候选向量的余弦相似度
    - 使用矩阵运算提高效率
    - 处理零向量情况
    
    Args:
        queries: 查询向量，形状为 (num_queries, dim) 的numpy数组
        candidates: 候选向量，形状为 (num_candidates, dim) 的numpy数组
        
    Returns:
        相似度矩阵，形状为 (num_queries, num_candidates) 的numpy数组
        每个元素 queries[i] 和 candidates[j] 的余弦相似度
    """
    # 确保输入是numpy数组
    queries = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    
    # 验证维度
    if queries.ndim != 2:
        raise ValueError(f"queries应该是2维数组，实际维度: {queries.ndim}")
    if candidates.ndim != 2:
        raise ValueError(f"candidates应该是2维数组，实际维度: {candidates.ndim}")
    
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            f"向量维度不匹配: queries.shape[1]={queries.shape[1]}, "
            f"candidates.shape[1]={candidates.shape[1]}"
        )
    
    # 计算点积矩阵: (num_queries, num_candidates)
    # queries @ candidates.T 得到 (num_queries, num_candidates)
    dot_products = np.dot(queries, candidates.T)
    
    # 计算每个查询向量的L2范数: (num_queries,)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    
    # 计算每个候选向量的L2范数: (num_candidates,)
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    
    # 计算范数乘积矩阵: (num_queries, num_candidates)
    norm_products = np.dot(query_norms, candidate_norms.T)
    
    # 处理零向量情况（将零向量对应的位置设为0）
    zero_mask = (query_norms == 0.0) | (candidate_norms.T == 0.0)
    
    # 计算余弦相似度矩阵
    # 避免除零错误：将norm_products为0的位置设为1（这样相似度会是0）
    norm_products_safe = np.where(norm_products == 0.0, 1.0, norm_products)
    similarities = dot_products / norm_products_safe
    
    # 将零向量对应的位置设为0
    similarities[zero_mask] = 0.0
    
    # 处理数值误差（确保结果在[-1, 1]范围内）
    similarities = np.clip(similarities, -1.0, 1.0)
    
    return similarities


