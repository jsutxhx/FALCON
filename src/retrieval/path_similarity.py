"""路径相似度计算模块"""
from typing import List
from src.knowledge_graph.graph_storage import KnowledgeGraph
from src.data_processing.data_structures import Paper


class PathSimilarityCalculator:
    """路径相似度计算器
    
    功能：
    - 计算知识图谱中路径的权重
    - 支持路径衰减因子
    - 用于计算两实体间的路径相似度
    
    Attributes:
        path_decay_factor: 路径衰减因子 η，默认为0.5
        graph: 知识图谱对象（可选，用于获取边权重）
    """
    
    def __init__(self, path_decay_factor: float = 0.5, graph: KnowledgeGraph = None):
        """初始化路径相似度计算器
        
        Args:
            path_decay_factor: 路径衰减因子 η，用于惩罚长路径，默认为0.5
            graph: 知识图谱对象，用于获取边权重。如果为None，需要在调用时提供
        """
        self.path_decay_factor = path_decay_factor
        self.graph = graph
    
    def _compute_path_weight(
        self,
        path: List[str],
        graph: KnowledgeGraph = None
    ) -> float:
        """计算单条路径的权重
        
        功能：
        - 计算路径权重：w(path) = Π w_r * η^L
        - 其中：
          - Π w_r: 路径上所有边的权重乘积
          - η: 路径衰减因子
          - L: 路径长度（边的数量）
        
        Args:
            path: 路径，节点ID列表，例如 ["node1", "node2", "node3"] 表示从node1到node2到node3的路径
            graph: 知识图谱对象，用于获取边权重。如果为None，使用self.graph
            
        Returns:
            路径权重值，范围在 [0, +∞) 之间（通常较小）
            如果路径无效（长度<2或边不存在），返回0.0
        """
        # 使用提供的graph或实例的graph
        if graph is None:
            graph = self.graph
        
        if graph is None:
            raise ValueError("graph must be provided either in __init__ or as parameter")
        
        # 路径至少需要2个节点（1条边）
        if len(path) < 2:
            return 0.0
        
        # 计算路径长度（边的数量）
        path_length = len(path) - 1
        
        # 计算路径上所有边的权重乘积
        edge_weight_product = 1.0
        
        for i in range(path_length):
            source = path[i]
            target = path[i + 1]
            
            # 检查边是否存在
            if not graph.graph.has_edge(source, target):
                # 如果边不存在，返回0.0
                return 0.0
            
            # 获取边权重
            edge_attrs = graph.graph[source][target]
            edge_weight = edge_attrs.get("weight", 1.0)
            
            # 累积乘积
            edge_weight_product *= edge_weight
        
        # 计算路径权重：w(path) = Π w_r * η^L
        path_weight = edge_weight_product * (self.path_decay_factor ** path_length)
        
        return float(path_weight)
    
    def _find_paths(
        self,
        entity1_id: str,
        entity2_id: str,
        max_length: int = 3,
        graph: KnowledgeGraph = None
    ) -> List[List[str]]:
        """枚举两个实体之间的所有路径
        
        功能：
        - 使用DFS（深度优先搜索）找到从entity1_id到entity2_id的所有路径
        - 限制最大路径长度（边的数量）
        - 路径中允许重复节点（但通常不会出现，因为限制了长度）
        
        Args:
            entity1_id: 起始实体ID
            entity2_id: 目标实体ID
            max_length: 最大路径长度（边的数量），默认为3
            graph: 知识图谱对象。如果为None，使用self.graph
            
        Returns:
            路径列表，每个路径是节点ID列表
            例如：[["e1", "e2"], ["e1", "e3", "e2"]] 表示找到两条路径
        """
        # 使用提供的graph或实例的graph
        if graph is None:
            graph = self.graph
        
        if graph is None:
            raise ValueError("graph must be provided either in __init__ or as parameter")
        
        # 检查节点是否存在
        if entity1_id not in graph.graph or entity2_id not in graph.graph:
            return []
        
        # 如果起始和目标相同，返回空路径列表（或可以返回[[entity1_id]]，但通常不需要）
        if entity1_id == entity2_id:
            return []
        
        # 存储所有找到的路径
        all_paths = []
        
        # 使用DFS递归搜索
        def dfs(current: str, target: str, path: List[str], remaining_length: int):
            """DFS递归函数
            
            Args:
                current: 当前节点ID
                target: 目标节点ID
                path: 当前路径（节点ID列表）
                remaining_length: 剩余可用的路径长度
            """
            # 如果到达目标节点，保存路径
            if current == target:
                all_paths.append(path[:])  # 复制路径
                return
            
            # 如果剩余长度为0，无法继续
            if remaining_length == 0:
                return
            
            # 遍历当前节点的所有出边
            if current in graph.graph:
                for neighbor in graph.graph[current]:
                    # 避免在路径中重复访问同一节点（防止循环，但允许在长路径中重复）
                    # 这里我们允许重复，但限制路径长度来避免无限循环
                    new_path = path + [neighbor]
                    dfs(neighbor, target, new_path, remaining_length - 1)
        
        # 从起始节点开始DFS
        dfs(entity1_id, entity2_id, [entity1_id], max_length)
        
        return all_paths
    
    def compute(
        self,
        query_paper: Paper,
        candidate_paper: Paper,
        graph: KnowledgeGraph,
        max_path_length: int = 3
    ) -> float:
        """计算两篇论文之间的路径相似度
        
        功能：
        - 枚举查询论文和候选论文实体之间的所有路径
        - 计算每条路径的权重并累加
        - 返回路径相似度总和
        
        算法：
        1. 从知识图谱中获取查询论文的所有实体（通过contains边）
        2. 从知识图谱中获取候选论文的所有实体（通过contains边）
        3. 对每对实体（查询论文的实体，候选论文的实体），枚举它们之间的路径
        4. 计算每条路径的权重并累加
        
        Args:
            query_paper: 查询论文对象
            candidate_paper: 候选论文对象
            graph: 知识图谱对象
            max_path_length: 最大路径长度（边的数量），默认为3
            
        Returns:
            路径相似度值，是所有路径权重的累加和
            如果两篇论文之间没有路径连接，返回0.0
        """
        # 使用提供的graph或实例的graph
        if graph is None:
            graph = self.graph
        
        if graph is None:
            raise ValueError("graph must be provided either as parameter or in __init__")
        
        # 检查论文节点是否存在
        if query_paper.id not in graph.graph:
            return 0.0
        if candidate_paper.id not in graph.graph:
            return 0.0
        
        # 获取查询论文的所有实体（通过contains边）
        query_entity_ids = []
        if query_paper.id in graph.graph:
            for neighbor in graph.graph[query_paper.id]:
                # 检查是否是contains边
                edge_attrs = graph.graph[query_paper.id][neighbor]
                if edge_attrs.get("relation_type") == "contains":
                    query_entity_ids.append(neighbor)
        
        # 获取候选论文的所有实体（通过contains边）
        candidate_entity_ids = []
        if candidate_paper.id in graph.graph:
            for neighbor in graph.graph[candidate_paper.id]:
                # 检查是否是contains边
                edge_attrs = graph.graph[candidate_paper.id][neighbor]
                if edge_attrs.get("relation_type") == "contains":
                    candidate_entity_ids.append(neighbor)
        
        # 如果没有实体，返回0.0
        if not query_entity_ids or not candidate_entity_ids:
            return 0.0
        
        # 累加所有路径的权重
        total_path_similarity = 0.0
        
        # 对每对实体枚举路径（限制实体对数量以提高性能）
        max_entity_pairs = 10  # 最多处理10对实体
        entity_pairs_processed = 0
        
        for query_entity_id in query_entity_ids[:5]:  # 限制查询实体数量
            if entity_pairs_processed >= max_entity_pairs:
                break
            for candidate_entity_id in candidate_entity_ids[:5]:  # 限制候选实体数量
                if entity_pairs_processed >= max_entity_pairs:
                    break
                
                # 枚举从查询实体到候选实体的所有路径（限制路径数量）
                paths = self._find_paths(
                    query_entity_id,
                    candidate_entity_id,
                    max_length=max_path_length,
                    graph=graph
                )
                
                # 只处理前5条路径（路径按长度排序，优先处理短路径）
                paths = paths[:5]
                
                # 计算每条路径的权重并累加
                for path in paths:
                    path_weight = self._compute_path_weight(path, graph=graph)
                    total_path_similarity += path_weight
                
                entity_pairs_processed += 1
        
        return float(total_path_similarity)

