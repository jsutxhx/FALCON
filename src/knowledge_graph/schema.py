"""知识图谱Schema定义模块"""
from enum import Enum


class EntityType(Enum):
    """实体类型枚举"""
    TASK = "task"
    METHOD = "method"
    MATERIAL = "material"
    METRIC = "metric"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, s: str) -> 'EntityType':
        """从字符串创建EntityType枚举
        
        Args:
            s: 实体类型字符串（不区分大小写）
            
        Returns:
            EntityType枚举值
            
        Raises:
            ValueError: 如果字符串不匹配任何枚举值
        """
        s_lower = s.lower()
        for entity_type in cls:
            if entity_type.value == s_lower:
                return entity_type
        raise ValueError(f"Unknown entity type: '{s}'. Must be one of: {[e.value for e in cls]}")


class RelationType(Enum):
    """关系类型枚举"""
    CONTAINS = "contains"
    HIERARCHY = "hierarchy"
    IMPLEMENT = "implement"
    USE = "use"
    EVALUATE = "evaluate"
    
    @classmethod
    def from_string(cls, s: str) -> 'RelationType':
        """从字符串创建RelationType枚举
        
        Args:
            s: 关系类型字符串（不区分大小写）
            
        Returns:
            RelationType枚举值
            
        Raises:
            ValueError: 如果字符串不匹配任何枚举值
        """
        s_lower = s.lower()
        for relation_type in cls:
            if relation_type.value == s_lower:
                return relation_type
        raise ValueError(f"Unknown relation type: '{s}'. Must be one of: {[e.value for e in cls]}")


class CitationFunction(Enum):
    """引用功能枚举"""
    BACKGROUND = "background"
    USE = "use"
    COMPARE = "compare"
    INSPIRE = "inspire"
    
    def __init__(self, value: str):
        """初始化枚举值并设置描述"""
        self._value_ = value
        # 为每个枚举值设置描述
        descriptions = {
            "background": "提供背景知识和相关工作",
            "use": "使用或采用的方法、技术或工具",
            "compare": "进行比较和对比",
            "inspire": "提供跨领域的启发和灵感"
        }
        self.description = descriptions.get(value, "")
    
    @classmethod
    def from_string(cls, s: str) -> 'CitationFunction':
        """从字符串创建CitationFunction枚举
        
        Args:
            s: 引用功能字符串（不区分大小写）
            
        Returns:
            CitationFunction枚举值
            
        Raises:
            ValueError: 如果字符串不匹配任何枚举值
        """
        s_lower = s.lower()
        for citation_function in cls:
            if citation_function.value == s_lower:
                return citation_function
        raise ValueError(f"Unknown citation function: '{s}'. Must be one of: {[e.value for e in cls]}")

