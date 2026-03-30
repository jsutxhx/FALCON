"""FAISS嵌入索引模块"""
import numpy as np
from typing import Tuple, Optional
try:
    import faiss
except ImportError:
    faiss = None


class EmbeddingIndex:
    """FAISS向量索引类
    
    功能：
    - 使用FAISS构建高效的向量检索索引
    - 支持快速相似度搜索
    
    Attributes:
        index: FAISS索引对象
        dimension: 向量维度
        is_built: 索引是否已构建
    """
    
    def __init__(self, dimension: int, metric: str = "L2"):
        """初始化嵌入索引
        
        Args:
            dimension: 向量维度
            metric: 距离度量方式，可选 "L2" 或 "IP"（内积）
                   "L2" 表示欧氏距离，"IP" 表示内积（用于余弦相似度）
        """
        if faiss is None:
            raise ImportError(
                "faiss is not installed. Please install it using: pip install faiss-cpu"
            )
        
        self.dimension = dimension
        self.metric = metric
        self.index: Optional[faiss.Index] = None
        self.is_built = False
        
        # 根据metric选择索引类型
        if metric == "L2":
            # 使用FlatL2索引（精确搜索）
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == "IP":
            # 使用FlatIP索引（内积，用于余弦相似度）
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Must be 'L2' or 'IP'")
    
    def build_index(self, embeddings: np.ndarray):
        """构建向量索引
        
        功能：
        - 将向量添加到FAISS索引中
        - 支持批量添加
        
        Args:
            embeddings: 向量数组，形状为 (n, dimension) 的numpy数组
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        # 确保embeddings是float32类型（FAISS要求）
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # 如果是IP（内积）索引，需要归一化向量（用于余弦相似度）
        if self.metric == "IP":
            # L2归一化
            faiss.normalize_L2(embeddings)
        
        # 添加向量到索引
        self.index.add(embeddings)
        self.is_built = True
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相似的k个向量
        
        功能：
        - 在索引中搜索与查询向量最相似的k个向量
        - 返回索引和距离
        
        Args:
            query: 查询向量，形状为 (dimension,) 或 (n_queries, dimension) 的numpy数组
            k: 返回的最相似向量数量
            
        Returns:
            (indices, distances) 元组:
            - indices: 最相似向量的索引，形状为 (n_queries, k) 的numpy数组
            - distances: 对应的距离，形状为 (n_queries, k) 的numpy数组
        """
        if not self.is_built:
            raise ValueError("Index has not been built. Call build_index() first.")
        
        # 处理单个查询向量
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query.shape[1]}"
            )
        
        # 确保query是float32类型
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        # 如果是IP（内积）索引，需要归一化查询向量
        if self.metric == "IP":
            faiss.normalize_L2(query)
        
        # 搜索
        distances, indices = self.index.search(query, k)
        
        return indices, distances


