"""数据结构定义模块"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Paper:
    """论文数据类
    
    Attributes:
        id: 论文唯一标识符
        title: 论文标题
        abstract: 论文摘要
        authors: 作者列表
        year: 发表年份
        venue: 发表期刊或会议
        citation_count: 引用次数
        doi: 数字对象标识符（可选）
    """
    id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    venue: str
    citation_count: int
    doi: Optional[str] = None
    
    def __repr__(self) -> str:
        """返回论文的字符串表示"""
        authors_str = ", ".join(self.authors[:3])  # 只显示前3个作者
        if len(self.authors) > 3:
            authors_str += f", et al. ({len(self.authors)} authors)"
        return (
            f"Paper(id='{self.id}', title='{self.title[:50]}...', "
            f"authors=[{authors_str}], year={self.year}, "
            f"venue='{self.venue}', citations={self.citation_count})"
        )


# 允许的实体类型
ALLOWED_ENTITY_TYPES = {"task", "method", "material", "metric", "other"}

# 允许的关系类型
ALLOWED_RELATION_TYPES = {"contains", "hierarchy", "implement", "use", "evaluate"}

# 允许的引用位置
ALLOWED_CITATION_POSITIONS = {"introduction", "methodology", "experiment", "discussion"}


@dataclass
class Entity:
    """知识实体数据类
    
    Attributes:
        id: 实体唯一标识符
        text: 实体文本
        entity_type: 实体类型 (task, method, material, metric, other)
        canonical: 规范化后的实体文本
        embedding: 实体嵌入向量（可选）
        weight: 实体权重
    """
    id: str
    text: str
    entity_type: str
    canonical: str
    embedding: Optional[np.ndarray] = None
    weight: float = 1.0
    
    def __post_init__(self):
        """验证entity_type是否为允许的值"""
        if self.entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(
                f"entity_type must be one of {ALLOWED_ENTITY_TYPES}, "
                f"got '{self.entity_type}'"
            )
    
    def __repr__(self) -> str:
        """返回实体的字符串表示"""
        embedding_info = f", embedding_shape={self.embedding.shape}" if self.embedding is not None else ""
        return (
            f"Entity(id='{self.id}', text='{self.text[:30]}...', "
            f"type='{self.entity_type}', canonical='{self.canonical}', "
            f"weight={self.weight}{embedding_info})"
        )


@dataclass
class Relation:
    """实体间关系数据类
    
    Attributes:
        head_id: 头实体ID
        tail_id: 尾实体ID
        relation_type: 关系类型 (contains, hierarchy, implement, use, evaluate)
        weight: 关系权重
    """
    head_id: str
    tail_id: str
    relation_type: str
    weight: float = 1.0
    
    def __post_init__(self):
        """验证relation_type是否为允许的值"""
        if self.relation_type not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"relation_type must be one of {ALLOWED_RELATION_TYPES}, "
                f"got '{self.relation_type}'"
            )
    
    def as_tuple(self) -> tuple:
        """返回三元组 (head, relation, tail)
        
        Returns:
            (head_id, relation_type, tail_id) 元组
        """
        return (self.head_id, self.relation_type, self.tail_id)
    
    def __repr__(self) -> str:
        """返回关系的字符串表示"""
        return (
            f"Relation(head='{self.head_id}', relation='{self.relation_type}', "
            f"tail='{self.tail_id}', weight={self.weight})"
        )


@dataclass
class Citation:
    """引用关系数据类
    
    Attributes:
        source_paper_id: 引用源论文ID
        target_paper_id: 被引用的目标论文ID
        context: 引用上下文文本
        position: 引用位置 (introduction, methodology, experiment, discussion)
    """
    source_paper_id: str
    target_paper_id: str
    context: str
    position: str
    
    def __post_init__(self):
        """验证position是否为允许的值"""
        if self.position not in ALLOWED_CITATION_POSITIONS:
            raise ValueError(
                f"position must be one of {ALLOWED_CITATION_POSITIONS}, "
                f"got '{self.position}'"
            )
    
    def __repr__(self) -> str:
        """返回引用的字符串表示"""
        context_preview = self.context[:50] + "..." if len(self.context) > 50 else self.context
        return (
            f"Citation(source='{self.source_paper_id}', target='{self.target_paper_id}', "
            f"position='{self.position}', context='{context_preview}')"
        )


@dataclass
class Recommendation:
    """推荐结果数据类
    
    Attributes:
        paper_id: 被推荐的论文ID
        score: 推荐得分
        reason: 推荐理由
        citation_position: 建议的引用位置 (introduction, methodology, experiment, discussion)
        confidence: 置信度 (high, medium, low)
    """
    paper_id: str
    score: float
    reason: str
    citation_position: str
    confidence: str
    
    def __post_init__(self):
        """验证字段值"""
        # 验证citation_position
        if self.citation_position not in ALLOWED_CITATION_POSITIONS:
            raise ValueError(
                f"citation_position must be one of {ALLOWED_CITATION_POSITIONS}, "
                f"got '{self.citation_position}'"
            )
        # 验证confidence
        allowed_confidence = {"high", "medium", "low"}
        if self.confidence not in allowed_confidence:
            raise ValueError(
                f"confidence must be one of {allowed_confidence}, "
                f"got '{self.confidence}'"
            )
    
    def to_dict(self) -> dict:
        """转换为字典格式
        
        Returns:
            包含所有字段的字典
        """
        return {
            "paper_id": self.paper_id,
            "score": self.score,
            "reason": self.reason,
            "citation_position": self.citation_position,
            "confidence": self.confidence
        }
    
    def __repr__(self) -> str:
        """返回推荐的字符串表示"""
        return (
            f"Recommendation(paper_id='{self.paper_id}', score={self.score:.4f}, "
            f"position='{self.citation_position}', confidence='{self.confidence}')"
        )


