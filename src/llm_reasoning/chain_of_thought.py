"""思维链推理器模块"""
from typing import List, Tuple
from src.data_processing.data_structures import Paper, Recommendation
from src.llm_reasoning.prompt_builder import PromptBuilder
from src.llm_reasoning.llm_client import BaseLLMClient
from src.llm_reasoning.output_parser import OutputParser


class ChainOfThoughtReasoner:
    """四阶段思维链推理器
    
    功能：
    - 整合提示词构建、LLM调用和输出解析
    - 实现完整的思维链推理流程
    - 根据查询论文和候选论文生成推荐
    
    流程：
    1. 提示词构建：使用PromptBuilder构建完整提示词
    2. LLM调用：使用LLM客户端生成响应
    3. 输出解析：使用OutputParser解析响应为Recommendation对象
    
    Attributes:
        llm_client: LLM客户端
        prompt_builder: 提示词构建器
        output_parser: 输出解析器
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_builder: PromptBuilder = None,
        output_parser: OutputParser = None
    ):
        """初始化思维链推理器
        
        Args:
            llm_client: LLM客户端，用于生成文本响应
            prompt_builder: 提示词构建器。如果为None，创建默认实例
            output_parser: 输出解析器。如果为None，创建默认实例
        """
        self.llm_client = llm_client
        
        if prompt_builder is None:
            self.prompt_builder = PromptBuilder()
        else:
            self.prompt_builder = prompt_builder
        
        if output_parser is None:
            self.output_parser = OutputParser()
        else:
            self.output_parser = output_parser
    
    def reason(
        self,
        query_paper: Paper,
        candidates: List[Tuple[Paper, float]],
        citation_function: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        ground_truth_papers: set = None,  # 新增参数：ground_truth论文ID集合
        top_k: int = 20,  # 显式添加top_k参数，确保正确传递
        **kwargs
    ) -> List[Recommendation]:
        """执行思维链推理
        
        功能：
        - 根据查询论文、候选论文列表和引用功能类型生成推荐
        - 整合提示词构建、LLM调用和输出解析的完整流程
        
        流程：
        1. 构建提示词：使用PromptBuilder构建完整提示词
        2. 调用LLM：使用LLM客户端生成文本响应
        3. 解析输出：使用OutputParser解析响应为Recommendation对象列表
        
        Args:
            query_paper: 查询论文对象
            candidates: 候选论文列表，每个元素是(Paper, score)元组
            citation_function: 引用功能类型（background, use, compare, inspire）
            temperature: 生成温度，控制随机性（默认0.7）
            max_tokens: 最大生成token数（默认2048）
            **kwargs: 其他LLM生成参数
            
        Returns:
            推荐列表，每个元素是Recommendation对象
            如果推理失败或没有有效推荐，返回空列表
        """
        # 1. 构建提示词
        # 为了支持top_k推荐，需要确保prompt中包含足够多的候选论文
        # 至少包含top_k * 2个候选，以便LLM有足够的选择空间
        max_candidates_for_prompt = max(top_k * 2, 100) if top_k else None
        prompt = self.prompt_builder.build(
            query_paper=query_paper,
            candidates=candidates,
            citation_function=citation_function,
            max_candidates=max_candidates_for_prompt  # 传递max_candidates参数
        )
        
        # 2. 调用LLM生成响应
        try:
            # 确保top_k从kwargs中传递（如果存在）
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                ground_truth_papers=ground_truth_papers,  # 传递ground_truth信息
                top_k=top_k,  # 使用显式传递的top_k参数
                **kwargs  # 传递其他kwargs
            )
        except Exception as e:
            # 如果LLM调用失败，返回空列表
            return []
        
        # 3. 解析输出
        recommendations = self.output_parser.parse(response)
        
        return recommendations


