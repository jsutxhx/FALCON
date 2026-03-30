"""文件工具函数模块"""
import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: str) -> None:
    """确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    """加载JSON文件
    
    Args:
        path: JSON文件路径
        
    Returns:
        解析后的字典对象
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """保存数据为JSON文件
    
    Args:
        data: 要保存的数据（可序列化为JSON）
        path: 保存路径
        indent: JSON缩进空格数，默认为2
    """
    path = Path(path)
    # 确保目录存在
    ensure_dir(str(path.parent))
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件（每行一个JSON对象）
    
    Args:
        path: JSONL文件路径
        
    Returns:
        字典列表
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """保存数据为JSONL文件（每行一个JSON对象）
    
    Args:
        data: 字典列表
        path: 保存路径
    """
    path = Path(path)
    # 确保目录存在
    ensure_dir(str(path.parent))
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')




