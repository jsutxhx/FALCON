"""数据加载器模块"""
from abc import ABC, abstractmethod
from typing import List
from .data_structures import Paper, Citation


class BaseDataLoader(ABC):
    """数据加载器抽象基类"""
    
    @abstractmethod
    def load_papers(self, path: str) -> List[Paper]:
        """加载论文数据
        
        Args:
            path: 数据文件路径
            
        Returns:
            论文对象列表
        """
        pass
    
    @abstractmethod
    def load_citations(self, path: str, paper_ids: set = None) -> List[Citation]:
        """加载引用关系数据
        
        Args:
            path: 数据文件路径
            paper_ids: 已加载的论文ID集合，用于验证source和target是否存在。
                      如果为None，则不进行验证
            
        Returns:
            引用对象列表
        """
        pass
    
    def _validate_paper(self, paper_data: dict) -> bool:
        """验证论文数据是否包含必需字段
        
        Args:
            paper_data: 论文数据字典
            
        Returns:
            如果包含所有必需字段返回True，否则返回False
        """
        required_fields = ["id", "title", "abstract", "authors", "year", "venue", "citation_count"]
        
        for field in required_fields:
            if field not in paper_data:
                return False
        
        # 验证字段类型
        if not isinstance(paper_data["id"], str):
            return False
        if not isinstance(paper_data["title"], str):
            return False
        if not isinstance(paper_data["abstract"], str):
            return False
        if not isinstance(paper_data["authors"], list):
            return False
        if not isinstance(paper_data["year"], int):
            return False
        if not isinstance(paper_data["venue"], str):
            return False
        if not isinstance(paper_data["citation_count"], int):
            return False
        
        return True


class JSONDataLoader(BaseDataLoader):
    """JSON格式数据加载器"""
    
    def __init__(self):
        """初始化JSON数据加载器"""
        from src.utils.file_utils import load_json
        self.load_json = load_json
    
    def load_papers(self, path: str) -> List[Paper]:
        """加载JSON格式的论文数据
        
        Args:
            path: JSON文件路径
            
        Returns:
            论文对象列表
        """
        data = self.load_json(path)
        
        # 如果数据是字典，尝试获取papers键
        if isinstance(data, dict):
            papers_data = data.get("papers", data.get("data", []))
        elif isinstance(data, list):
            papers_data = data
        else:
            papers_data = []
        
        papers = []
        for paper_data in papers_data:
            # 验证数据
            if not self._validate_paper(paper_data):
                continue
            
            # 处理缺失字段的默认值
            paper = Paper(
                id=paper_data["id"],
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                authors=paper_data.get("authors", []),
                year=paper_data.get("year", 0),
                venue=paper_data.get("venue", ""),
                citation_count=paper_data.get("citation_count", 0),
                doi=paper_data.get("doi", None)
            )
            papers.append(paper)
        
        return papers
    
    def load_citations(self, path: str, paper_ids: set = None) -> List[Citation]:
        """加载JSON格式的引用关系数据
        
        Args:
            path: JSON文件路径
            paper_ids: 已加载的论文ID集合，用于验证source和target是否存在。
                      如果为None，则不进行验证
            
        Returns:
            引用对象列表
        """
        from src.data_processing.data_structures import Citation
        
        data = self.load_json(path)
        
        # 如果数据是字典，尝试获取citations键
        if isinstance(data, dict):
            citations_data = data.get("citations", data.get("data", []))
        elif isinstance(data, list):
            citations_data = data
        else:
            citations_data = []
        
        citations = []
        for citation_data in citations_data:
            # 验证必需字段
            required_fields = ["source_paper_id", "target_paper_id", "context", "position"]
            if not all(field in citation_data for field in required_fields):
                continue
            
            # 验证source和target论文ID是否存在
            if paper_ids is not None:
                source_id = citation_data["source_paper_id"]
                target_id = citation_data["target_paper_id"]
                if source_id not in paper_ids or target_id not in paper_ids:
                    continue
            
            try:
                citation = Citation(
                    source_paper_id=citation_data["source_paper_id"],
                    target_paper_id=citation_data["target_paper_id"],
                    context=citation_data["context"],
                    position=citation_data["position"]
                )
                citations.append(citation)
            except ValueError:
                # 跳过无效的position值
                continue
        
        return citations

