"""配置加载器模块"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """配置类，支持点访问"""
    
    def __init__(self, data: Dict[str, Any]):
        """初始化配置对象
        
        Args:
            data: 配置字典
        """
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """支持点访问"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持默认值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值，如果不存在则返回默认值
        """
        return self._data.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._data
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config({self._data})"


def load_config(config_path: str) -> Config:
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config对象，支持点访问
        
    Example:
        >>> config = load_config("config/base_config.yaml")
        >>> print(config.project_name)
        >>> print(config.data.raw_dir)
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    return Config(data)




