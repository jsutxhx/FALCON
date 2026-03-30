"""设备管理工具模块"""
import torch
from typing import Union, Any


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """获取设备对象，自动检测CUDA可用性
    
    Args:
        device: 设备名称 ("cuda", "cpu") 或 torch.device 对象。
                如果为None，则自动检测CUDA可用性
    
    Returns:
        torch.device对象
    
    Example:
        >>> device = get_device()  # 自动检测
        >>> device = get_device("cuda")
        >>> device = get_device("cpu")
    """
    if device is None:
        # 自动检测CUDA可用性
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # 如果已经是torch.device对象，直接返回
    if isinstance(device, torch.device):
        return device
    
    # 转换为torch.device对象
    return torch.device(device)


def move_to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """将对象移动到指定设备
    
    支持的对象类型：
    - torch.Tensor
    - torch.nn.Module
    - dict（递归移动值）
    - list/tuple（递归移动元素）
    - 其他类型直接返回
    
    Args:
        obj: 要移动的对象
        device: 目标设备
    
    Returns:
        移动到目标设备后的对象
    
    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> device = get_device()
        >>> tensor = move_to_device(tensor, device)
    """
    device = get_device(device)
    
    # 处理Tensor
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    
    # 处理Module
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    
    # 处理字典（递归移动值）
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    
    # 处理列表（递归移动元素）
    if isinstance(obj, (list, tuple)):
        moved_list = [move_to_device(item, device) for item in obj]
        # 保持原类型（list或tuple）
        return type(obj)(moved_list)
    
    # 其他类型直接返回
    return obj




