"""日志工具模块"""
import sys
from pathlib import Path
from loguru import logger

# 跟踪已配置的日志文件，避免重复添加
_configured_files = set()
_console_configured = False


def get_logger(name: str, log_dir: str = "logs") -> logger:
    """获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称（通常是模块名）
        log_dir: 日志文件目录，默认为 "logs"
        
    Returns:
        配置好的 loguru logger 实例
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a test message")
    """
    global _console_configured, _configured_files
    
    # 日志格式: 时间、级别、模块、消息
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # 只添加一次控制台输出
    if not _console_configured:
        logger.remove()  # 移除默认处理器
        logger.add(
            sys.stderr,
            format=log_format,
            level="INFO",
            colorize=True
        )
        _console_configured = True
    
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名使用模块名（清理模块名中的点号）
    log_name = name.replace(".", "_")
    log_file = log_path / f"{log_name}.log"
    
    # 只为每个文件添加一次文件输出处理器
    if str(log_file) not in _configured_files:
        logger.add(
            str(log_file),
            format=log_format,
            level="DEBUG",  # 文件中记录更详细的日志
            rotation="10 MB",  # 日志文件大小达到10MB时轮转
            retention="7 days",  # 保留7天的日志
            encoding="utf-8"
        )
        _configured_files.add(str(log_file))
    
    return logger

