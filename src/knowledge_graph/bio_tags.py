"""BIO标签定义模块"""
from typing import Dict, Optional
from src.knowledge_graph.schema import EntityType


# BIO标签列表
# 格式: O, B-{EntityType}, I-{EntityType}
# 共 1 + 5*2 = 11 个标签
BIO_LABELS = [
    "O",                    # 0: Outside (非实体)
    "B-Task",              # 1: Begin-Task
    "I-Task",              # 2: Inside-Task
    "B-Method",            # 3: Begin-Method
    "I-Method",            # 4: Inside-Method
    "B-Material",          # 5: Begin-Material
    "I-Material",          # 6: Inside-Material
    "B-Metric",            # 7: Begin-Metric
    "I-Metric",            # 8: Inside-Metric
    "B-Other",             # 9: Begin-Other
    "I-Other",             # 10: Inside-Other
]

# 标签到ID的映射
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}

# ID到标签的映射
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(BIO_LABELS)}

# 标签数量
NUM_LABELS = len(BIO_LABELS)


def label_to_id(label: str) -> int:
    """将BIO标签转换为ID
    
    Args:
        label: BIO标签字符串（如 "B-Task", "I-Method", "O"）
        
    Returns:
        标签对应的ID
        
    Raises:
        ValueError: 如果标签不存在
    """
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown BIO label: '{label}'. Must be one of: {BIO_LABELS}")
    return LABEL_TO_ID[label]


def id_to_label(label_id: int) -> str:
    """将ID转换为BIO标签
    
    Args:
        label_id: 标签ID
        
    Returns:
        标签字符串
        
    Raises:
        ValueError: 如果ID超出范围
    """
    if label_id < 0 or label_id >= NUM_LABELS:
        raise ValueError(f"Label ID out of range: {label_id}. Must be in [0, {NUM_LABELS})")
    return ID_TO_LABEL[label_id]


def get_entity_type_from_label(label: str) -> Optional[EntityType]:
    """从BIO标签中提取实体类型
    
    功能：
    - 从BIO标签（如 "B-Task", "I-Method"）中提取实体类型
    - 如果标签是 "O"，返回 None
    
    Args:
        label: BIO标签字符串
        
    Returns:
        实体类型枚举值，如果是 "O" 则返回 None
        
    Raises:
        ValueError: 如果标签格式不正确
    """
    if label == "O":
        return None
    
    # 解析标签格式: B-{Type} 或 I-{Type}
    if "-" not in label:
        raise ValueError(f"Invalid BIO label format: '{label}'. Expected format: 'B-Type' or 'I-Type'")
    
    parts = label.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid BIO label format: '{label}'. Expected format: 'B-Type' or 'I-Type'")
    
    prefix, entity_type_str = parts
    
    # 验证前缀
    if prefix not in ["B", "I"]:
        raise ValueError(f"Invalid BIO prefix: '{prefix}'. Must be 'B' or 'I'")
    
    # 转换为EntityType枚举
    try:
        entity_type = EntityType.from_string(entity_type_str)
        return entity_type
    except ValueError:
        raise ValueError(f"Unknown entity type in label '{label}': '{entity_type_str}'")

