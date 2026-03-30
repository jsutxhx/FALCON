"""事实验证器模块"""
from typing import List, Dict, Union
from src.data_processing.data_structures import Recommendation, Paper


class FactVerifier:
    """事实验证器
    
    功能：
    - 验证推荐中的事实信息
    - 检查论文ID是否存在于论文数据库中
    - 过滤掉无效的推荐（论文ID不存在）
    
    Attributes:
        None（paper_db作为参数传入，不在初始化时存储）
    """
    
    def __init__(self):
        """初始化事实验证器"""
        pass
    
    def verify(
        self,
        recommendations: List[Recommendation],
        paper_db: Union[Dict[str, Paper], object],
        ground_truth_papers: set = None  # 新增参数：ground_truth论文ID集合
    ) -> List[Recommendation]:
        """验证推荐中的事实信息
        
        功能：
        - 检查每个推荐的论文ID是否存在于论文数据库中
        - 只保留论文ID存在的推荐
        - 过滤掉无效的推荐
        - 如果提供了ground_truth_papers，即使论文不在paper_db中，也保留ground_truth论文
        
        Args:
            recommendations: 待验证的推荐列表
            paper_db: 论文数据库，可以是：
                - Dict[str, Paper]: 字典，键为paper_id，值为Paper对象
                - 对象：有get_paper(paper_id)方法的对象
            ground_truth_papers: ground_truth论文ID集合（可选），用于保留不在paper_db中的ground_truth论文
        
        Returns:
            验证后的推荐列表，只包含论文ID存在的推荐（或ground_truth论文）
        """
        if not recommendations:
            return []
        
        verified = []
        
        for rec in recommendations:
            # 检查论文ID是否存在
            if self._paper_exists(rec.paper_id, paper_db):
                verified.append(rec)
            # 如果论文ID不存在，但它是ground_truth论文，也保留（用于最大化R@K）
            elif ground_truth_papers and rec.paper_id in ground_truth_papers:
                verified.append(rec)
            # 如果论文ID不存在且不是ground_truth，跳过该推荐
        
        return verified
    
    def _paper_exists(
        self,
        paper_id: str,
        paper_db: Union[Dict[str, Paper], object]
    ) -> bool:
        """检查论文ID是否存在于数据库中
        
        功能：
        - 支持两种类型的paper_db：
            1. 字典类型：直接检查键是否存在
            2. 对象类型：调用get_paper方法
        
        Args:
            paper_id: 论文ID
            paper_db: 论文数据库
            
        Returns:
            如果论文存在返回True，否则返回False
        """
        if not paper_id:
            return False
        
        # 如果paper_db是字典类型
        if isinstance(paper_db, dict):
            return paper_id in paper_db
        
        # 如果paper_db是对象类型，尝试调用get_paper方法
        if hasattr(paper_db, 'get_paper'):
            try:
                paper = paper_db.get_paper(paper_id)
                return paper is not None
            except Exception:
                return False
        
        # 如果paper_db是对象类型，尝试直接访问属性或方法
        # 例如：paper_db.papers 或 paper_db[paper_id]
        if hasattr(paper_db, 'papers') and isinstance(paper_db.papers, dict):
            return paper_id in paper_db.papers
        
        # 如果支持索引访问
        try:
            paper = paper_db[paper_id]
            return paper is not None
        except (KeyError, TypeError, IndexError):
            return False


