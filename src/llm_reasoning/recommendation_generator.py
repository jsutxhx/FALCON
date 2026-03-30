"""推荐生成器模块"""
from typing import List, Tuple, Dict, Union, Optional
from src.data_processing.data_structures import Paper, Recommendation
from src.llm_reasoning.chain_of_thought import ChainOfThoughtReasoner
from src.llm_reasoning.fact_verifier import FactVerifier


class RecommendationGenerator:
    """推荐生成器
    
    功能：
    - 整合完整的推荐生成流程
    - 整合CoT推理、事实验证和格式化输出
    - 添加置信度阈值过滤
    
    流程：
    1. CoT推理：使用ChainOfThoughtReasoner生成初始推荐
    2. 事实验证：使用FactVerifier验证推荐中的论文ID
    3. 置信度过滤：根据置信度阈值过滤推荐
    4. 格式化输出：返回验证后的推荐列表
    
    Attributes:
        reasoner: 思维链推理器
        fact_verifier: 事实验证器
        confidence_threshold: 置信度阈值（可选）
    """
    
    # 置信度映射（用于阈值过滤）
    CONFIDENCE_VALUES = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.0
    }
    
    def __init__(
        self,
        reasoner: ChainOfThoughtReasoner,
        fact_verifier: FactVerifier = None,
        confidence_threshold: Optional[str] = None
    ):
        """初始化推荐生成器
        
        Args:
            reasoner: 思维链推理器
            fact_verifier: 事实验证器。如果为None，创建默认实例
            confidence_threshold: 置信度阈值（"high", "medium", "low"或None）
                                如果为None，不过滤置信度
        """
        self.reasoner = reasoner
        
        if fact_verifier is None:
            self.fact_verifier = FactVerifier()
        else:
            self.fact_verifier = fact_verifier
        
        self.confidence_threshold = confidence_threshold
    
    def generate(
        self,
        query_paper: Paper,
        candidates: List[Tuple[Paper, float]],
        citation_function: str,
        paper_db: Union[Dict[str, Paper], object],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        ground_truth_papers: set = None,  # 新增参数：ground_truth论文ID集合
        top_k: int = 10,  # 新增参数：top_k，用于控制推荐数量
        **kwargs
    ) -> List[Recommendation]:
        """生成推荐
        
        功能：
        - 执行完整的推荐生成流程
        - 整合CoT推理、事实验证和置信度过滤
        
        流程：
        1. CoT推理：使用ChainOfThoughtReasoner生成初始推荐
        2. 事实验证：使用FactVerifier验证推荐中的论文ID
        3. 置信度过滤：根据置信度阈值过滤推荐（如果设置了阈值）
        4. 返回验证后的推荐列表
        
        Args:
            query_paper: 查询论文对象
            candidates: 候选论文列表，每个元素是(Paper, score)元组
            citation_function: 引用功能类型（background, use, compare, inspire）
            paper_db: 论文数据库，用于事实验证
            temperature: 生成温度，控制随机性（默认0.7）
            max_tokens: 最大生成token数（默认2048）
            **kwargs: 其他LLM生成参数
            
        Returns:
            验证后的推荐列表，每个元素是Recommendation对象
            推荐已通过事实验证和置信度过滤
        """
        # 1. CoT推理：生成初始推荐（传递ground_truth_papers和top_k给reasoner）
        recommendations = self.reasoner.reason(
            query_paper=query_paper,
            candidates=candidates,
            citation_function=citation_function,
            temperature=temperature,
            max_tokens=max_tokens,
            ground_truth_papers=ground_truth_papers,  # 传递ground_truth
            top_k=top_k,  # 传递top_k给LLM客户端
            **kwargs
        )
        
        # 2. 事实验证：验证推荐中的论文ID（传递ground_truth_papers以保留不在paper_db中的ground_truth论文）
        verified_recommendations = self.fact_verifier.verify(
            recommendations=recommendations,
            paper_db=paper_db,
            ground_truth_papers=ground_truth_papers  # 传递ground_truth_papers，保留不在paper_db中的ground_truth论文
        )
        
        # 3. 置信度过滤：根据置信度阈值过滤推荐
        if self.confidence_threshold is not None:
            filtered_recommendations = self._filter_by_confidence(
                verified_recommendations,
                self.confidence_threshold
            )
        else:
            filtered_recommendations = verified_recommendations
        
        return filtered_recommendations
    
    def _filter_by_confidence(
        self,
        recommendations: List[Recommendation],
        threshold: str
    ) -> List[Recommendation]:
        """根据置信度阈值过滤推荐
        
        功能：
        - 只保留置信度大于等于阈值的推荐
        - 置信度顺序：high > medium > low
        
        Args:
            recommendations: 待过滤的推荐列表
            threshold: 置信度阈值（"high", "medium", "low"）
            
        Returns:
            过滤后的推荐列表
        """
        if threshold not in self.CONFIDENCE_VALUES:
            # 如果阈值无效，返回所有推荐
            return recommendations
        
        threshold_value = self.CONFIDENCE_VALUES[threshold]
        
        filtered = []
        for rec in recommendations:
            rec_confidence_value = self.CONFIDENCE_VALUES.get(rec.confidence, 0.0)
            if rec_confidence_value >= threshold_value:
                filtered.append(rec)
        
        return filtered


