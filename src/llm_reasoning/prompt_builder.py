"""提示词构建器模块"""
from typing import List, Tuple
from src.data_processing.data_structures import Paper
from src.llm_reasoning.prompt_templates import RECOMMENDATION_PROMPT, get_function_description


class PromptBuilder:
    """提示词构建器
    
    功能：
    - 根据查询论文、候选论文列表和引用功能类型构建完整的提示词
    - 格式化候选论文信息，使其易于LLM理解
    - 使用预定义的提示词模板
    
    Attributes:
        template: 提示词模板字符串
    """
    
    def __init__(self, template: str = None):
        """初始化提示词构建器
        
        Args:
            template: 提示词模板字符串。如果为None，使用默认的RECOMMENDATION_PROMPT
        """
        self.template = template if template is not None else RECOMMENDATION_PROMPT
    
    def build(
        self,
        query_paper: Paper,
        candidates: List[Tuple[Paper, float]],
        citation_function: str,
        max_candidates: int = None  # 新增参数：限制候选数量，如果为None则不限制
    ) -> str:
        """构建完整的提示词
        
        功能：
        - 格式化查询论文信息
        - 格式化候选论文列表
        - 使用模板构建完整提示词
        
        Args:
            query_paper: 查询论文对象
            candidates: 候选论文列表，每个元素是(Paper, score)元组
            citation_function: 引用功能类型（background, use, compare, inspire）
            max_candidates: 限制候选数量，如果为None则不限制（默认None）
            
        Returns:
            完整的提示词字符串
        """
        # 格式化候选论文列表
        candidate_list = self._format_candidates(candidates, max_candidates=max_candidates)
        
        # 使用模板构建提示词
        prompt = self.template.format(
            query_title=query_paper.title,
            query_abstract=query_paper.abstract,
            citation_function=citation_function,
            candidate_list=candidate_list
        )
        
        return prompt
    
    def _format_candidates(
        self,
        candidates: List[Tuple[Paper, float]],
        max_candidates: int = None  # 新增参数：限制候选数量，如果为None则不限制
    ) -> str:
        """格式化候选论文列表
        
        功能：
        - 将候选论文列表格式化为易读的文本格式
        - 包含论文的关键信息：标题、作者、年份、venue、引用次数、得分
        - 按序号排列
        
        Args:
            candidates: 候选论文列表，每个元素是(Paper, score)元组
            max_candidates: 限制候选数量，如果为None则不限制（默认None）
            
        Returns:
            格式化后的候选论文列表字符串
        """
        if not candidates:
            return "No candidate papers provided."
        
        lines = []
        seen_ids = set()  # 用于去重，确保每个论文只出现一次
        for i, (paper, score) in enumerate(candidates, 1):
            # 跳过重复的论文ID（提高多样性）
            if paper.id in seen_ids:
                continue
            seen_ids.add(paper.id)
            
            # 格式化作者列表（最多显示3个作者）
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += f", et al. ({len(paper.authors)} authors)"
            
            # 构建论文信息
            paper_info = f"{len(seen_ids)}. Paper ID: {paper.id}\n"
            paper_info += f"   Title: {paper.title}\n"
            paper_info += f"   Authors: {authors_str}\n"
            paper_info += f"   Year: {paper.year}\n"
            paper_info += f"   Venue: {paper.venue}\n"
            paper_info += f"   Citations: {paper.citation_count}\n"
            
            # 添加摘要（如果存在且不太长）
            if paper.abstract:
                # 限制摘要长度，避免提示词过长
                abstract_preview = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                paper_info += f"   Abstract: {abstract_preview}\n"
            
            # 添加得分信息
            paper_info += f"   Relevance Score: {score:.4f}"
            
            lines.append(paper_info)
            
            # 限制候选数量，避免提示词过长
            # 如果max_candidates为None，不限制；否则限制为max_candidates
            # 为了支持top_k=50，至少需要包含50个候选（或更多，以便LLM有足够的选择空间）
            limit = max_candidates if max_candidates is not None else float('inf')
            if len(seen_ids) >= limit:
                break
        
        return "\n\n".join(lines)


