"""认知评分器模块"""
from typing import Dict, List
from src.data_processing.data_structures import Paper, Entity
from src.retrieval.entity_similarity import EntitySimilarityCalculator
from src.retrieval.path_similarity import PathSimilarityCalculator
from src.reranking.function_scorer import FunctionScorer
from src.knowledge_graph.graph_storage import KnowledgeGraph


class CognitiveScorer:
    """认知导向评分器
    
    功能：
    - 综合实体相似度、路径相似度和功能评分计算认知导向得分
    - 公式: S_cog = S_entity + S_path + S_function
    其中：
    - S_entity: 实体相似度（Entity Similarity）
    - S_path: 路径相似度（Path Similarity）
    - S_function: 功能特定评分（Function-specific Score）
    
    Attributes:
        entity_calculator: 实体相似度计算器
        path_calculator: 路径相似度计算器
        function_scorer: 功能评分器
        graph: 知识图谱对象（用于获取论文实体）
    """
    
    def __init__(
        self,
        entity_calculator: EntitySimilarityCalculator,
        path_calculator: PathSimilarityCalculator,
        function_scorer: FunctionScorer,
        graph: KnowledgeGraph
    ):
        """初始化认知评分器
        
        Args:
            entity_calculator: 实体相似度计算器
            path_calculator: 路径相似度计算器
            function_scorer: 功能评分器
            graph: 知识图谱对象，用于获取论文实体
        """
        self.entity_calculator = entity_calculator
        self.path_calculator = path_calculator
        self.function_scorer = function_scorer
        self.graph = graph
        # 添加实体缓存以提高性能
        self._entity_cache = {}
    
    def _get_paper_entities(self, paper: Paper) -> Dict[str, List[Entity]]:
        """从知识图谱中获取论文的所有实体（带缓存）
        
        功能：
        - 通过contains边从知识图谱中获取论文的所有实体
        - 按实体类型分组
        - 使用缓存避免重复计算
        
        Args:
            paper: 论文对象
            
        Returns:
            实体字典，键为实体类型，值为该类型的实体列表
        """
        # 检查缓存
        if paper.id in self._entity_cache:
            return self._entity_cache[paper.id]
        
        entities_by_type: Dict[str, List[Entity]] = {}
        
        # 检查论文节点是否存在
        if paper.id not in self.graph.graph:
            self._entity_cache[paper.id] = entities_by_type
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
                        # 获取embedding（如果存在）
                        embedding = None
                        if "embedding" in entity_node and entity_node["embedding"] is not None:
                            embedding_list = entity_node["embedding"]
                            embedding_shape = entity_node.get("embedding_shape")
                            if embedding_shape:
                                import numpy as np
                                embedding = np.array(embedding_list).reshape(embedding_shape)
                            else:
                                import numpy as np
                                embedding = np.array(embedding_list)
                        
                        # 创建Entity对象
                        entity = Entity(
                            id=entity_node.get("id", neighbor_id),
                            text=entity_node.get("text", ""),
                            entity_type=entity_type,
                            canonical=entity_node.get("canonical", ""),
                            weight=entity_node.get("weight", 1.0),
                            embedding=embedding
                        )
                        
                        # 按类型分组
                        if entity_type not in entities_by_type:
                            entities_by_type[entity_type] = []
                        entities_by_type[entity_type].append(entity)
        
        # 缓存结果
        self._entity_cache[paper.id] = entities_by_type
        return entities_by_type
    
    def score(
        self,
        query_paper: Paper,
        candidate_paper: Paper,
        function: str
    ) -> float:
        """计算认知导向评分
        
        功能：
        - 综合实体相似度、路径相似度和功能评分
        - 公式: S_cog = S_entity + S_path + S_function
        
        其中：
        - S_entity: 实体相似度，通过EntitySimilarityCalculator计算
        - S_path: 路径相似度，通过PathSimilarityCalculator计算
        - S_function: 功能特定评分，根据function类型调用FunctionScorer的相应方法
        
        Args:
            query_paper: 查询论文对象
            candidate_paper: 候选论文对象
            function: 引用功能类型，可选值："background", "use", "compare", "inspire"
            
        Returns:
            认知导向评分，理论上范围在[0, +∞)之间（各项相加）
        """
        # 1. 计算实体相似度 S_entity
        query_entities = self._get_paper_entities(query_paper)
        candidate_entities = self._get_paper_entities(candidate_paper)
        
        s_entity = self.entity_calculator.compute(query_entities, candidate_entities)
        
        # 2. 计算路径相似度 S_path（限制路径长度和数量以提高性能）
        s_path = self.path_calculator.compute(
            query_paper, 
            candidate_paper, 
            self.graph,
            max_path_length=2  # 减少最大路径长度以提高性能
        )
        
        # 3. 计算功能特定评分 S_function
        function_lower = function.lower()
        if function_lower == "background":
            s_function = self.function_scorer.score_background(query_paper, candidate_paper)
        elif function_lower == "use":
            s_function = self.function_scorer.score_use(query_paper, candidate_paper)
        elif function_lower == "compare":
            s_function = self.function_scorer.score_compare(query_paper, candidate_paper)
        elif function_lower == "inspire":
            s_function = self.function_scorer.score_inspire(query_paper, candidate_paper)
        else:
            # 未知功能类型，使用background作为默认
            s_function = self.function_scorer.score_background(query_paper, candidate_paper)
        
        # 4. 综合评分
        s_cog = s_entity + s_path + s_function
        
        return s_cog


