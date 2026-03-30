"""功能评分器模块"""
from typing import Dict, List, Optional
import numpy as np
from src.data_processing.data_structures import Paper, Entity
from src.retrieval.entity_similarity import EntitySimilarityCalculator
from src.retrieval.similarity_utils import cosine_similarity
from src.embeddings.specter_encoder import SpecterEncoder
from src.knowledge_graph.graph_storage import KnowledgeGraph


class FunctionScorer:
    """功能特定评分器
    
    功能：
    - 根据不同的引用功能计算特定的评分
    - Background功能：关注领域匹配度（TS + DS + PS）
    - Use功能：关注技术匹配度（MS + ES + AS）
    - Compare功能：强调可比性和方法差异
    - Inspire功能：强调跨领域价值
    
    Attributes:
        entity_calculator: 实体相似度计算器
        specter_encoder: SPECTER编码器（用于摘要相似度计算）
        graph: 知识图谱对象（用于获取论文实体）
    """
    
    def __init__(
        self,
        entity_calculator: EntitySimilarityCalculator,
        graph: KnowledgeGraph,
        specter_encoder: Optional[SpecterEncoder] = None
    ):
        """初始化功能评分器
        
        Args:
            entity_calculator: 实体相似度计算器
            graph: 知识图谱对象，用于获取论文实体
            specter_encoder: SPECTER编码器，用于计算摘要相似度。如果为None，会在需要时创建
        """
        self.entity_calculator = entity_calculator
        self.graph = graph
        self.specter_encoder = specter_encoder
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
                                embedding = np.array(embedding_list).reshape(embedding_shape)
                            else:
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
    
    def _task_similarity(self, p1: Paper, p2: Paper) -> float:
        """任务实体相似度
        
        功能：
        - 获取两篇论文的task类型实体
        - 计算task实体之间的相似度
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            任务相似度，范围在[0, 1]之间
        """
        # 获取论文的实体
        entities1 = self._get_paper_entities(p1)
        entities2 = self._get_paper_entities(p2)
        
        # 获取task类型实体
        task_entities1 = entities1.get("task", [])
        task_entities2 = entities2.get("task", [])
        
        # 如果任一论文没有task实体，返回0
        if not task_entities1 or not task_entities2:
            return 0.0
        
        # 使用entity_calculator计算task类型实体的相似度
        task_entities_dict1 = {"task": task_entities1}
        task_entities_dict2 = {"task": task_entities2}
        
        # 创建一个临时的EntitySimilarityCalculator，只关注task类型
        from src.retrieval.entity_similarity import EntitySimilarityCalculator
        temp_calculator = EntitySimilarityCalculator(type_weights={"task": 1.0})
        
        similarity = temp_calculator._compute_intra_type_similarity(task_entities1, task_entities2)
        
        return similarity
    
    def _domain_similarity(self, p1: Paper, p2: Paper) -> float:
        """领域相似度
        
        功能：
        - 使用venue的Jaccard相似度作为领域相似度的近似
        - 如果venue相同，返回1.0；否则返回0.0
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            领域相似度，范围在[0, 1]之间
        """
        # 简化实现：使用venue的匹配作为领域相似度
        # 如果venue相同，认为领域相似；否则不相似
        if p1.venue and p2.venue:
            # 简单的字符串匹配（不区分大小写）
            if p1.venue.lower() == p2.venue.lower():
                return 1.0
        
        return 0.0
    
    def _abstract_similarity(self, p1: Paper, p2: Paper) -> float:
        """摘要相似度（优化版本：离线模式下使用简化计算）
        
        功能：
        - 使用SPECTER编码器编码两篇论文的标题和摘要
        - 计算编码向量的余弦相似度
        - 在离线模式下，使用基于标题和摘要文本的简单相似度
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            摘要相似度，范围在[-1, 1]之间（通常为[0, 1]）
        """
        # 检查是否离线模式（通过检查环境变量或编码器状态）
        import os
        offline_mode = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        
        # 如果离线模式或编码器不可用，使用简化的文本相似度
        if offline_mode or self.specter_encoder is None:
            # 使用简单的文本重叠度作为近似
            title_words1 = set((p1.title or "").lower().split())
            title_words2 = set((p2.title or "").lower().split())
            abstract_words1 = set((p1.abstract or "").lower().split()[:50])  # 只取前50个词
            abstract_words2 = set((p2.abstract or "").lower().split()[:50])
            
            # 计算Jaccard相似度
            title_overlap = len(title_words1 & title_words2) / max(len(title_words1 | title_words2), 1)
            abstract_overlap = len(abstract_words1 & abstract_words2) / max(len(abstract_words1 | abstract_words2), 1)
            
            # 加权平均
            similarity = 0.4 * title_overlap + 0.6 * abstract_overlap
            return float(similarity)
        
        # 如果specter_encoder未提供，尝试创建临时实例
        encoder = self.specter_encoder
        if encoder is None:
            try:
                encoder = SpecterEncoder()
            except Exception:
                # 如果无法创建编码器（例如网络问题），返回0
                return 0.0
        
        # 编码两篇论文
        try:
            embedding1 = encoder.encode_paper(p1.title, p1.abstract)
            embedding2 = encoder.encode_paper(p2.title, p2.abstract)
            
            # 计算余弦相似度
            similarity = cosine_similarity(embedding1, embedding2)
            
            return similarity
        except Exception:
            # 如果编码失败（例如网络问题），返回0
            return 0.0
    
    def score_background(self, query_paper: Paper, candidate_paper: Paper) -> float:
        """计算Background功能评分
        
        功能：
        - Background功能关注领域匹配度
        - 公式: S_bg = TS + DS + PS
        其中：
        - TS: Task Similarity（任务相似度）
        - DS: Domain Similarity（领域相似度）
        - PS: Abstract Similarity（摘要相似度）
        
        Args:
            query_paper: 查询论文
            candidate_paper: 候选论文
            
        Returns:
            Background功能评分，理论上范围在[0, 3]之间（TS, DS, PS各在[0, 1]范围内）
        """
        ts = self._task_similarity(query_paper, candidate_paper)
        ds = self._domain_similarity(query_paper, candidate_paper)
        ps = self._abstract_similarity(query_paper, candidate_paper)
        
        return ts + ds + ps
    
    def _method_similarity(self, p1: Paper, p2: Paper) -> float:
        """方法实体相似度
        
        功能：
        - 获取两篇论文的method类型实体
        - 计算method实体之间的相似度
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            方法相似度，范围在[0, 1]之间
        """
        # 获取论文的实体
        entities1 = self._get_paper_entities(p1)
        entities2 = self._get_paper_entities(p2)
        
        # 获取method类型实体
        method_entities1 = entities1.get("method", [])
        method_entities2 = entities2.get("method", [])
        
        # 如果任一论文没有method实体，返回0
        if not method_entities1 or not method_entities2:
            return 0.0
        
        # 使用entity_calculator计算method类型实体的相似度
        from src.retrieval.entity_similarity import EntitySimilarityCalculator
        temp_calculator = EntitySimilarityCalculator(type_weights={"method": 1.0})
        
        similarity = temp_calculator._compute_intra_type_similarity(method_entities1, method_entities2)
        
        return similarity
    
    def _metric_overlap(self, p1: Paper, p2: Paper) -> float:
        """指标实体重叠度
        
        功能：
        - 获取两篇论文的metric类型实体
        - 计算metric实体的Jaccard重叠度
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            指标重叠度，范围在[0, 1]之间（Jaccard相似度）
        """
        # 获取论文的实体
        entities1 = self._get_paper_entities(p1)
        entities2 = self._get_paper_entities(p2)
        
        # 获取metric类型实体
        metric_entities1 = entities1.get("metric", [])
        metric_entities2 = entities2.get("metric", [])
        
        # 如果任一论文没有metric实体，返回0
        if not metric_entities1 or not metric_entities2:
            return 0.0
        
        # 获取canonical形式的metric实体集合
        m1 = set(e.canonical for e in metric_entities1 if e.canonical)
        m2 = set(e.canonical for e in metric_entities2 if e.canonical)
        
        if not m1 or not m2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(m1 & m2)
        union = len(m1 | m2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _material_overlap(self, p1: Paper, p2: Paper) -> float:
        """材料实体重叠度
        
        功能：
        - 获取两篇论文的material类型实体
        - 计算material实体的Jaccard重叠度
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            材料重叠度，范围在[0, 1]之间（Jaccard相似度）
        """
        # 获取论文的实体
        entities1 = self._get_paper_entities(p1)
        entities2 = self._get_paper_entities(p2)
        
        # 获取material类型实体
        material_entities1 = entities1.get("material", [])
        material_entities2 = entities2.get("material", [])
        
        # 如果任一论文没有material实体，返回0
        if not material_entities1 or not material_entities2:
            return 0.0
        
        # 获取canonical形式的material实体集合
        mat1 = set(e.canonical for e in material_entities1 if e.canonical)
        mat2 = set(e.canonical for e in material_entities2 if e.canonical)
        
        if not mat1 or not mat2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(mat1 & mat2)
        union = len(mat1 | mat2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def score_use(self, query_paper: Paper, candidate_paper: Paper) -> float:
        """计算Use功能评分
        
        功能：
        - Use功能关注技术匹配度
        - 公式: S_use = MS + ES + AS
        其中：
        - MS: Method Similarity（方法相似度）
        - ES: Metric Overlap（指标重叠度）
        - AS: Material Overlap（材料重叠度）
        
        Args:
            query_paper: 查询论文
            candidate_paper: 候选论文
            
        Returns:
            Use功能评分，理论上范围在[0, 3]之间（MS, ES, AS各在[0, 1]范围内）
        """
        ms = self._method_similarity(query_paper, candidate_paper)
        es = self._metric_overlap(query_paper, candidate_paper)
        as_ = self._material_overlap(query_paper, candidate_paper)
        
        return ms + es + as_
    
    def _experimental_comparability(self, p1: Paper, p2: Paper) -> float:
        """实验可比性
        
        功能：
        - 衡量两篇论文的实验是否可比
        - 基于任务相似度、指标重叠度和材料重叠度
        - 可比实验应该具有相似的任务、指标和材料
        
        Args:
            p1: 论文1
            p2: 论文2
            
        Returns:
            实验可比性，范围在[0, 1]之间
        """
        # 任务相似度：可比实验应该有相似的任务
        ts = self._task_similarity(p1, p2)
        
        # 指标重叠度：可比实验应该使用相似的评估指标
        es = self._metric_overlap(p1, p2)
        
        # 材料重叠度：可比实验可能使用相似的数据集或材料
        as_ = self._material_overlap(p1, p2)
        
        # 综合计算：取平均值作为实验可比性
        # 如果任务、指标、材料都相似，则实验可比性高
        comparability = (ts + es + as_) / 3.0
        
        return comparability
    
    def _method_advantage(self, p1: Paper, p2: Paper) -> float:
        """方法优势/抽象度
        
        功能：
        - 衡量候选论文的方法是否具有跨领域应用的优势
        - 对于Inspire功能，我们希望找到方法抽象度高、可跨领域应用的方法
        - 基于方法相似度，但考虑方法的通用性
        
        Args:
            p1: 查询论文
            p2: 候选论文
            
        Returns:
            方法优势，范围在[0, 1]之间
        """
        # 获取两篇论文的方法实体
        entities1 = self._get_paper_entities(p1)
        entities2 = self._get_paper_entities(p2)
        
        method_entities1 = entities1.get("method", [])
        method_entities2 = entities2.get("method", [])
        
        # 如果任一论文没有method实体，返回0
        if not method_entities1 or not method_entities2:
            return 0.0
        
        # 计算方法的相似度
        # 对于Inspire功能，我们希望方法有一定的相似性（可以借鉴），
        # 但不需要完全相同（需要跨领域创新）
        from src.retrieval.entity_similarity import EntitySimilarityCalculator
        temp_calculator = EntitySimilarityCalculator(type_weights={"method": 1.0})
        
        similarity = temp_calculator._compute_intra_type_similarity(method_entities1, method_entities2)
        
        # 方法优势：中等相似度的方法更有跨领域应用价值
        # 如果完全相似，则没有创新价值；如果完全不相似，则难以借鉴
        # 这里我们使用相似度本身，表示方法的可借鉴性
        return similarity
    
    def score_compare(self, query_paper: Paper, candidate_paper: Paper) -> float:
        """计算Compare功能评分
        
        功能：
        - Compare功能强调可比性和方法差异
        - 公式: S_com = EC + (1 - MS)
        其中：
        - EC: Experimental Comparability（实验可比性）
        - MS: Method Similarity（方法相似度）
        - (1 - MS): 方法差异度，方法越不同，差异度越高
        
        说明：
        - Compare功能用于比较不同方法在相同任务上的表现
        - 需要实验可比（EC高），但方法不同（MS低，即1-MS高）
        
        Args:
            query_paper: 查询论文
            candidate_paper: 候选论文
            
        Returns:
            Compare功能评分，理论上范围在[0, 2]之间（EC和1-MS各在[0, 1]范围内）
        """
        ec = self._experimental_comparability(query_paper, candidate_paper)
        ms = self._method_similarity(query_paper, candidate_paper)
        
        return ec + (1.0 - ms)
    
    def score_inspire(self, query_paper: Paper, candidate_paper: Paper) -> float:
        """计算Inspire功能评分
        
        功能：
        - Inspire功能强调跨领域价值
        - 公式: S_ins = (1 - DS) + MA
        其中：
        - DS: Domain Similarity（领域相似度）
        - (1 - DS): 领域差异度，领域越不同，差异度越高
        - MA: Method Advantage（方法优势），表示方法的可借鉴性
        
        说明：
        - Inspire功能用于寻找跨领域的启发
        - 需要领域不同（DS低，即1-DS高），但方法可借鉴（MA高）
        
        Args:
            query_paper: 查询论文
            candidate_paper: 候选论文
            
        Returns:
            Inspire功能评分，理论上范围在[0, 2]之间（1-DS和MA各在[0, 1]范围内）
        """
        ds = self._domain_similarity(query_paper, candidate_paper)
        ma = self._method_advantage(query_paper, candidate_paper)
        
        return (1.0 - ds) + ma

