"""质量评分器模块"""
from typing import Dict
import numpy as np
from datetime import datetime
from src.data_processing.data_structures import Paper


class QualityScorer:
    """质量导向评分器
    
    功能：
    - 根据论文的质量指标计算评分
    - 考虑引用次数、作者h-index、期刊/会议排名、发表时间
    - 公式: S_qual = log(1 + cite) + h-index + venue_rank + 1/(1 + years)
    
    Attributes:
        venue_rankings: 期刊/会议排名字典
        current_year: 当前年份（用于计算发表年限）
    """
    
    def __init__(self, current_year: int = None):
        """初始化质量评分器
        
        Args:
            current_year: 当前年份，如果为None则使用系统当前年份
        """
        self.current_year = current_year if current_year is not None else datetime.now().year
        self.venue_rankings = self._init_venue_rankings()
    
    def _init_venue_rankings(self) -> Dict[str, float]:
        """初始化期刊/会议排名字典
        
        Returns:
            期刊/会议名称到排名的映射字典，排名范围[0, 1]
        """
        rankings = {
            # 顶级期刊
            "nature": 1.0,
            "science": 1.0,
            "cell": 0.98,
            "lancet": 0.98,
            "nejm": 0.98,
            
            # 顶级AI/ML会议
            "neurips": 0.95,
            "nips": 0.95,  # 旧名称
            "icml": 0.95,
            "iclr": 0.95,
            "aaai": 0.90,
            "ijcai": 0.90,
            
            # 顶级CV会议
            "cvpr": 0.95,
            "iccv": 0.93,
            "eccv": 0.93,
            
            # 顶级NLP会议
            "acl": 0.92,
            "emnlp": 0.90,
            "naacl": 0.90,
            
            # 顶级系统会议
            "osdi": 0.95,
            "sosp": 0.95,
            "usenix": 0.90,
            
            # 顶级数据库会议
            "sigmod": 0.93,
            "vldb": 0.93,
            "icde": 0.88,
            
            # 其他知名期刊/会议
            "pnas": 0.92,
            "plos one": 0.70,
            "arxiv": 0.50,  # 预印本
        }
        
        # 创建不区分大小写的版本
        normalized_rankings = {}
        for key, value in rankings.items():
            normalized_key = key.lower().replace(" ", "").replace("-", "")
            normalized_rankings[normalized_key] = value
        
        return normalized_rankings
    
    def _normalize(self, value: float, min_val: float = 0, max_val: float = 100) -> float:
        """Min-Max归一化到[0,1]
        
        Args:
            value: 待归一化的值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            归一化后的值，范围[0, 1]
        """
        if max_val <= min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        return max(0.0, min(1.0, normalized))  # 限制在[0, 1]范围内
    
    def _avg_author_hindex(self, paper: Paper) -> float:
        """计算作者平均h-index
        
        注意：
        - 由于Paper类中authors是字符串列表，没有h_index属性
        - 这里返回0.0作为默认值
        - 在实际应用中，可以通过外部数据源获取作者的h-index
        
        Args:
            paper: 论文对象
            
        Returns:
            作者平均h-index，当前实现返回0.0
        """
        # 由于Paper.authors是List[str]，没有h_index属性
        # 在实际应用中，可以通过外部API或数据库获取作者的h-index
        # 这里返回0.0作为占位符
        return 0.0
    
    def _venue_rank(self, paper: Paper) -> float:
        """期刊/会议排名分数
        
        功能：
        - 根据venue名称查找对应的排名
        - 如果未找到，返回默认值0.5
        
        Args:
            paper: 论文对象
            
        Returns:
            期刊/会议排名分数，范围[0, 1]
        """
        if not paper.venue:
            return 0.5
        
        # 规范化venue名称（转小写、去除空格和连字符）
        venue_key = paper.venue.lower().replace(" ", "").replace("-", "")
        
        # 查找排名
        rank = self.venue_rankings.get(venue_key, 0.5)
        
        return rank
    
    def _years_since_publication(self, paper: Paper) -> int:
        """计算论文发表至今的年数
        
        Args:
            paper: 论文对象
            
        Returns:
            发表年数，非负整数
        """
        years = self.current_year - paper.year
        return max(0, years)
    
    def score(self, paper: Paper) -> float:
        """计算论文质量评分
        
        功能：
        - 综合多个质量指标计算评分
        - 公式: S_qual = log(1 + cite) + h-index + venue_rank + 1/(1 + years)
        
        其中：
        - log(1 + cite): 引用次数的对数，使用log1p避免log(0)
        - h-index: 作者平均h-index（归一化到[0,1]）
        - venue_rank: 期刊/会议排名（已在[0,1]范围内）
        - 1/(1 + years): 时效性分数，越新分数越高
        
        Args:
            paper: 论文对象
            
        Returns:
            质量评分，理论上范围在[0, 4]左右（各项相加）
        """
        # 1. 引用次数分数：log(1 + citation_count)
        citation_score = np.log1p(paper.citation_count)
        
        # 2. 作者h-index分数（归一化到[0,1]）
        hindex_raw = self._avg_author_hindex(paper)
        hindex_score = self._normalize(hindex_raw, min_val=0, max_val=100)
        
        # 3. 期刊/会议排名分数（已在[0,1]范围内）
        venue_score = self._venue_rank(paper)
        
        # 4. 时效性分数：1/(1 + years_since_publication)
        years = self._years_since_publication(paper)
        recency_score = 1.0 / (1.0 + years)
        
        # 综合评分
        total_score = citation_score + hindex_score + venue_score + recency_score
        
        return total_score


