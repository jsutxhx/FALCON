"""PyTorch Dataset 模块"""
from torch.utils.data import Dataset
from typing import List, Dict, Any
import torch
from src.data_processing.data_structures import Paper


class PaperDataset(Dataset):
    """论文数据集类
    
    功能：
    - 存储论文列表
    - 支持索引访问
    - 支持长度查询
    
    Attributes:
        papers: 论文列表
    """
    
    def __init__(self, papers: List[Paper]):
        """初始化数据集
        
        Args:
            papers: 论文列表
        """
        self.papers = papers
    
    def __len__(self) -> int:
        """返回数据集大小
        
        Returns:
            论文数量
        """
        return len(self.papers)
    
    def __getitem__(self, idx: int) -> Paper:
        """获取指定索引的论文
        
        Args:
            idx: 论文索引
            
        Returns:
            Paper对象
            
        Raises:
            IndexError: 如果索引超出范围
        """
        if idx < 0 or idx >= len(self.papers):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.papers)})")
        return self.papers[idx]


def collate_papers(batch: List[Paper]) -> Dict[str, Any]:
    """将Paper列表转换为批次字典
    
    功能：
    - 将Paper对象列表转换为字典格式
    - 处理变长文本字段（title, abstract）
    - 将数值字段转换为tensor
    
    Args:
        batch: Paper对象列表
        
    Returns:
        包含以下字段的字典:
        - ids: 论文ID列表
        - titles: 标题列表
        - abstracts: 摘要列表
        - authors: 作者列表（每个元素是作者列表）
        - years: 年份tensor
        - venues: 期刊/会议列表
        - citation_counts: 引用次数tensor
        - dois: DOI列表（可能包含None）
    """
    if not batch:
        return {
            "ids": [],
            "titles": [],
            "abstracts": [],
            "authors": [],
            "years": torch.tensor([], dtype=torch.long),
            "venues": [],
            "citation_counts": torch.tensor([], dtype=torch.long),
            "dois": []
        }
    
    # 提取各个字段
    ids = [paper.id for paper in batch]
    titles = [paper.title for paper in batch]
    abstracts = [paper.abstract for paper in batch]
    authors = [paper.authors for paper in batch]
    years = torch.tensor([paper.year for paper in batch], dtype=torch.long)
    venues = [paper.venue for paper in batch]
    citation_counts = torch.tensor([paper.citation_count for paper in batch], dtype=torch.long)
    dois = [paper.doi for paper in batch]
    
    return {
        "ids": ids,
        "titles": titles,
        "abstracts": abstracts,
        "authors": authors,
        "years": years,
        "venues": venues,
        "citation_counts": citation_counts,
        "dois": dois
    }

