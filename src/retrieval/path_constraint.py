"""路径约束模块"""
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union, Dict
from src.knowledge_graph.graph_storage import KnowledgeGraph


class PathConstraint:
    """实体类型路径约束
    
    功能：
    - 定义允许的路径模式（实体类型 -> 关系类型 -> 实体类型）
    - 修改转移矩阵，将不满足约束的路径概率设为0
    - 重新归一化矩阵，使每行和为1
    
    Attributes:
        ALLOWED_PATTERNS: 允许的路径模式列表
            每个模式是一个三元组 (source_entity_type, relation_type, target_entity_type)
    """
    
    # 允许的路径模式
    ALLOWED_PATTERNS = [
        ("task", "implement", "method"),
        ("method", "use", "material"),
        ("metric", "evaluate", "method"),
        ("task", "hierarchy", "task"),
        ("method", "hierarchy", "method"),
    ]
    
    def apply(
        self,
        transition_matrix: Union[np.ndarray, sp.csr_matrix],
        node_types: List[str],
        edge_types: Union[np.ndarray, sp.csr_matrix, Dict[Tuple[int, int], str]]
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """应用路径约束到转移矩阵
        
        功能：
        - 检查每个转移 (i, j) 是否满足允许的路径模式
        - 如果不满足，将转移概率设为0
        - 重新归一化矩阵，使每行和为1
        
        算法：
        1. 对于每个非零转移 M[i, j]：
           - 检查模式 (node_types[i], edge_types[i, j], node_types[j]) 是否在 ALLOWED_PATTERNS 中
           - 如果不在，将 M[i, j] 设为0
        2. 重新归一化：M'[i, :] = M[i, :] / sum(M[i, :])
        
        Args:
            transition_matrix: 转移矩阵（可以是numpy数组或稀疏矩阵）
            node_types: 节点类型列表，长度为n，node_types[i]是节点i的实体类型
            edge_types: 边类型，可以是：
                - numpy数组或稀疏矩阵：edge_types[i, j]是边(i, j)的关系类型
                - 字典：{(i, j): relation_type}，键是边的索引对，值是关系类型字符串
        
        Returns:
            约束后的转移矩阵（与输入类型相同：numpy数组或稀疏矩阵）
        """
        # 转换为密集矩阵以便处理（如果是稀疏矩阵）
        is_sparse = sp.issparse(transition_matrix)
        if is_sparse:
            M = transition_matrix.toarray()
        else:
            M = transition_matrix.copy()
        
        n = len(node_types)
        assert M.shape[0] == n and M.shape[1] == n, (
            f"转移矩阵形状 {M.shape} 与节点数量 {n} 不匹配"
        )
        
        # 处理边类型
        edge_type_matrix = self._process_edge_types(edge_types, n)
        
        # 创建约束后的矩阵
        constrained_M = M.copy()
        
        # 遍历所有非零转移
        for i in range(n):
            for j in range(n):
                if M[i, j] > 0:
                    # 获取节点类型和边类型
                    source_type = node_types[i]
                    target_type = node_types[j]
                    relation_type = edge_type_matrix[i, j]
                    
                    # 如果边类型为空，跳过（可能是无效边）
                    if relation_type is None or relation_type == "":
                        constrained_M[i, j] = 0
                        continue
                    
                    # 检查路径模式是否允许
                    pattern = (source_type, relation_type, target_type)
                    if pattern not in self.ALLOWED_PATTERNS:
                        constrained_M[i, j] = 0
        
        # 重新归一化：每行和为1
        row_sums = constrained_M.sum(axis=1, keepdims=True)
        # 避免除零错误：如果某行全为0，保持为0
        row_sums = np.maximum(row_sums, 1e-10)
        constrained_M = constrained_M / row_sums
        
        # 如果输入是稀疏矩阵，转换回稀疏矩阵
        if is_sparse:
            constrained_M = sp.csr_matrix(constrained_M)
        
        return constrained_M
    
    def _process_edge_types(
        self,
        edge_types: Union[np.ndarray, sp.csr_matrix, Dict[Tuple[int, int], str]],
        n: int
    ) -> np.ndarray:
        """处理边类型输入，转换为矩阵格式
        
        Args:
            edge_types: 边类型输入（可以是数组、稀疏矩阵或字典）
            n: 节点数量
            
        Returns:
            边类型矩阵（numpy数组），edge_type_matrix[i, j]是边(i, j)的关系类型字符串
        """
        if isinstance(edge_types, dict):
            # 字典格式：{(i, j): relation_type}
            edge_type_matrix = np.full((n, n), None, dtype=object)
            for (i, j), relation_type in edge_types.items():
                if 0 <= i < n and 0 <= j < n:
                    edge_type_matrix[i, j] = relation_type
        elif sp.issparse(edge_types):
            # 稀疏矩阵：需要转换为密集矩阵（注意：稀疏矩阵不能存储字符串）
            # 这种情况下，假设edge_types存储的是关系类型的索引或编码
            # 为了简化，我们假设输入是字典格式
            raise ValueError("稀疏矩阵格式的edge_types不支持，请使用字典或密集数组")
        else:
            # numpy数组
            edge_type_matrix = np.asarray(edge_types)
            assert edge_type_matrix.shape == (n, n), (
                f"边类型矩阵形状 {edge_type_matrix.shape} 与节点数量 {n} 不匹配"
            )
        
        return edge_type_matrix


