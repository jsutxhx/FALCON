"""重排序器主类模块"""
from typing import List, Tuple
from src.data_processing.data_structures import Paper
from src.reranking.citation_function_classifier import CitationFunctionClassifier
from src.reranking.function_scorer import FunctionScorer
from src.reranking.cognitive_scorer import CognitiveScorer
from src.reranking.quality_scorer import QualityScorer
from src.reranking.weight_fusion import DynamicWeightFusion


class CitationReranker:
    """引用重排序主类
    
    功能：
    - 整合所有重排序组件，实现完整的重排序流程
    - 根据引用上下文和候选论文列表，返回重排序后的论文列表
    - 公式: S_final = α_cog · S_cog + α_qual · S_qual
    
    流程：
    1. 分类引用功能（通过CitationFunctionClassifier）
    2. 获取动态权重（通过DynamicWeightFusion）
    3. 计算每个候选的最终得分
       - 认知导向得分：S_cog = CognitiveScorer.score()
       - 质量导向得分：S_qual = QualityScorer.score()
       - 融合得分：S_final = α_cog · S_cog + α_qual · S_qual
    4. 按得分降序排列
    
    Attributes:
        function_classifier: 引用功能分类器
        function_scorer: 功能评分器
        cognitive_scorer: 认知评分器
        quality_scorer: 质量评分器
        weight_fusion: 动态权重融合网络
    """
    
    def __init__(
        self,
        function_classifier: CitationFunctionClassifier,
        function_scorer: FunctionScorer,
        cognitive_scorer: CognitiveScorer,
        quality_scorer: QualityScorer,
        weight_fusion: DynamicWeightFusion
    ):
        """初始化引用重排序器
        
        Args:
            function_classifier: 引用功能分类器，用于识别引用功能类型
            function_scorer: 功能评分器，用于计算功能特定评分
            cognitive_scorer: 认知评分器，用于计算认知导向得分
            quality_scorer: 质量评分器，用于计算质量导向得分
            weight_fusion: 动态权重融合网络，用于预测权重
        """
        self.function_classifier = function_classifier
        self.function_scorer = function_scorer
        self.cognitive_scorer = cognitive_scorer
        self.quality_scorer = quality_scorer
        self.weight_fusion = weight_fusion
    
    def rerank(
        self,
        query_paper: Paper,
        candidates: List[Paper],
        citation_context: str,
        max_candidates: int = None
    ) -> List[Tuple[Paper, float]]:
        """重排序候选论文
        
        功能：
        - 根据引用上下文和查询论文，对候选论文列表进行重排序
        - 返回按得分降序排列的论文列表
        
        流程：
        1. 分类引用功能：使用CitationFunctionClassifier识别引用功能类型
        2. 获取动态权重：使用DynamicWeightFusion根据功能类型预测权重
        3. 计算每个候选的最终得分：
           - S_cog = CognitiveScorer.score(query_paper, candidate, function)
           - S_qual = QualityScorer.score(candidate)
           - S_final = α_cog · S_cog + α_qual · S_qual
        4. 按得分降序排列
        
        Args:
            query_paper: 查询论文对象
            candidates: 候选论文列表
            citation_context: 引用上下文文本，用于分类引用功能
            max_candidates: 最大处理候选数量（用于性能优化），如果为None则处理所有候选
            
        Returns:
            重排序后的论文列表，每个元素是(Paper, score)元组
            列表按得分降序排列（得分高的在前）
        """
        # 限制候选数量以优化性能
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        
        # 1. 分类引用功能
        function, _ = self.function_classifier.classify(citation_context)
        
        # 2. 获取动态权重
        alpha_cog, alpha_qual = self.weight_fusion.forward(function)
        
        # 3. 批量计算质量得分（质量得分不依赖于query_paper，可以批量计算）
        quality_scores = {}
        for candidate in candidates:
            quality_scores[candidate.id] = self.quality_scorer.score(candidate)
        
        # 4. 计算每个候选的最终得分
        scored_candidates = []
        for candidate in candidates:
            # 认知导向得分
            s_cog = self.cognitive_scorer.score(query_paper, candidate, function)
            
            # 质量导向得分（从缓存中获取）
            s_qual = quality_scores[candidate.id]
            
            # 融合得分
            s_final = alpha_cog * s_cog + alpha_qual * s_qual
            
            scored_candidates.append((candidate, s_final))
        
        # 5. 按得分降序排列
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates


