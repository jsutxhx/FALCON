"""重启随机游走模块"""
import scipy.sparse as sp
import numpy as np
from typing import List, Optional
from src.knowledge_graph.graph_storage import KnowledgeGraph


class RestartedRandomWalk:
    """重启随机游走（Restarted Random Walk, RWR）
    
    功能：
    - 从图的邻接矩阵构建转移矩阵
    - 执行重启随机游走迭代
    - 用于计算节点间的相似度
    
    Attributes:
        adjacency_matrix: 邻接矩阵（稀疏矩阵）
        transition_matrix: 转移矩阵 M = D^{-1}A
        restart_prob: 重启概率 c，默认为0.15
        max_iterations: 最大迭代次数，默认为20
        convergence_threshold: 收敛阈值，默认为1e-6
    """
    
    def __init__(
        self,
        adjacency_matrix: Optional[sp.csr_matrix] = None,
        graph: Optional[KnowledgeGraph] = None,
        restart_prob: float = 0.15,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-6
    ):
        """初始化重启随机游走
        
        Args:
            adjacency_matrix: 邻接矩阵（稀疏矩阵）。如果为None，则从graph构建
            graph: 知识图谱对象。如果adjacency_matrix为None，则使用此参数构建邻接矩阵
            restart_prob: 重启概率 c，默认为0.15
            max_iterations: 最大迭代次数，默认为20
            convergence_threshold: 收敛阈值，默认为1e-6
        """
        self.restart_prob = restart_prob
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # 构建或获取邻接矩阵
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        elif graph is not None:
            self.adjacency_matrix = self._build_adjacency_matrix(graph)
        else:
            raise ValueError("Either adjacency_matrix or graph must be provided")
        
        # 构建转移矩阵 M = D^{-1}A
        self.transition_matrix = self._build_transition_matrix()
    
    def _build_adjacency_matrix(self, graph: KnowledgeGraph) -> sp.csr_matrix:
        """从知识图谱构建邻接矩阵
        
        功能：
        - 将KnowledgeGraph转换为稀疏邻接矩阵
        - 矩阵元素为边的权重
        
        Args:
            graph: 知识图谱对象
            
        Returns:
            稀疏邻接矩阵（CSR格式）
        """
        # 获取所有节点ID并建立索引映射
        nodes = list(graph.graph.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}
        n = len(nodes)
        
        # 构建邻接矩阵（使用边的权重）
        row_indices = []
        col_indices = []
        data = []
        
        for source, target, attrs in graph.graph.edges(data=True):
            if source in node_to_idx and target in node_to_idx:
                row_idx = node_to_idx[source]
                col_idx = node_to_idx[target]
                weight = attrs.get("weight", 1.0)
                
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                data.append(weight)
        
        # 创建稀疏矩阵
        adjacency_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n, n)
        )
        
        return adjacency_matrix
    
    def _build_transition_matrix(self) -> sp.csr_matrix:
        """构建转移矩阵 M = D^{-1}A
        
        功能：
        - 计算度矩阵的逆 D^{-1}
        - 计算转移矩阵 M = D^{-1}A
        - 转移矩阵每行和为1（行归一化）
        
        Returns:
            转移矩阵（稀疏矩阵，CSR格式）
        """
        # 计算每个节点的出度（行和）
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        
        # 处理零度节点（避免除零错误）
        # 对于零度节点，度设为1（这样D^{-1} = 1，但后续会处理）
        degrees = np.maximum(degrees, 1e-10)
        
        # 构建度矩阵的逆 D^{-1}（对角矩阵）
        D_inv = sp.diags(1.0 / degrees, format='csr')
        
        # 计算转移矩阵 M = D^{-1}A
        transition_matrix = D_inv @ self.adjacency_matrix
        
        return transition_matrix
    
    def run(self, seed_nodes: List[int]) -> np.ndarray:
        """执行重启随机游走迭代
        
        功能：
        - 从种子节点开始执行重启随机游走
        - 迭代公式: p^{k+1} = c * M^T * p^k + (1-c) * p^0
        - 检查收敛条件
        
        算法：
        1. 初始化分布 p^0：在种子节点位置均匀分布
        2. 迭代更新：p^{k+1} = c * M^T * p^k + (1-c) * p^0
        3. 检查收敛：||p^{k+1} - p^k|| < threshold
        4. 返回稳态分布
        
        Args:
            seed_nodes: 种子节点索引列表（整数列表，对应邻接矩阵的行/列索引）
            
        Returns:
            稳态分布向量，形状为 (n,) 的numpy数组，其中n是节点数量
            向量和应该等于1（概率分布）
        """
        n = self.adjacency_matrix.shape[0]
        
        # 验证种子节点索引
        for seed in seed_nodes:
            if seed < 0 or seed >= n:
                raise ValueError(f"Seed node index {seed} is out of range [0, {n-1}]")
        
        if not seed_nodes:
            raise ValueError("seed_nodes cannot be empty")
        
        # 初始化分布 p^0：在种子节点位置均匀分布
        p0 = np.zeros(n, dtype=np.float64)
        p0[seed_nodes] = 1.0 / len(seed_nodes)
        
        # 初始化当前分布
        p = p0.copy()
        
        # 迭代更新
        for iteration in range(self.max_iterations):
            # 计算 p^{k+1} = (1-c) * M^T * p^k + c * p^0
            # 其中 c 是重启概率（restart_prob）
            # (1-c) 是继续游走的概率
            # M^T 是转移矩阵的转置
            p_new = (
                (1 - self.restart_prob) * (self.transition_matrix.T @ p) +
                self.restart_prob * p0
            )
            
            # 检查收敛条件
            diff = np.linalg.norm(p_new - p)
            if diff < self.convergence_threshold:
                p = p_new
                break
            
            p = p_new
        
        # 归一化以确保概率分布和为1（处理数值误差）
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum
        else:
            # 如果所有概率都为0（不应该发生），返回初始分布
            p = p0.copy()
        
        return p

