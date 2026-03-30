"""OpenCorpus数据集加载器模块"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from .data_structures import Paper, Citation
from .data_loader import BaseDataLoader
from src.utils.file_utils import load_json, load_jsonl


class OpenCorpusLoader(BaseDataLoader):
    """OpenCorpus数据集加载器
    
    功能：
    - 加载OpenCorpus格式的论文数据
    - OpenCorpus是计算机科学领域的学术论文数据集
    - 包含约70,000篇论文，约1.2M条引用，时间跨度2000-2018
    
    数据格式：
    - 论文数据：JSON或JSONL格式
    - 引用数据：JSON或JSONL格式，包含source、target、context等字段
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """初始化OpenCorpus加载器
        
        Args:
            data_dir: 数据目录路径，如果为None，需要在使用时提供完整路径
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def load_papers(
        self,
        path: Optional[str] = None,
        max_papers: Optional[int] = None
    ) -> List[Paper]:
        """加载OpenCorpus格式的论文数据
        
        功能：
        - 支持JSON和JSONL格式
        - 自动检测文件格式
        - 支持限制加载的论文数量（用于快速测试）
        
        Args:
            path: 论文数据文件路径。如果为None且self.data_dir存在，尝试使用默认路径
            max_papers: 最大加载论文数量，用于快速测试。如果为None，加载所有论文
            
        Returns:
            论文对象列表
        """
        if path is None:
            if self.data_dir:
                # 尝试默认路径
                possible_paths = [
                    self.data_dir / "papers.json",
                    self.data_dir / "papers.jsonl",
                    self.data_dir / "opencorpus_papers.json",
                    self.data_dir / "opencorpus_papers.jsonl"
                ]
                for p in possible_paths:
                    if p.exists():
                        path = str(p)
                        break
                if path is None:
                    raise FileNotFoundError(f"未找到论文数据文件，请检查: {self.data_dir}")
            else:
                raise ValueError("必须提供论文数据文件路径")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"论文数据文件不存在: {path}")
        
        # 根据文件扩展名判断格式
        papers_data = []
        if path.suffix == ".jsonl":
            papers_data = load_jsonl(str(path))
        else:
            data = load_json(str(path))
            # 如果数据是字典，尝试获取papers键
            if isinstance(data, dict):
                papers_data = data.get("papers", data.get("data", data.get("items", [])))
            elif isinstance(data, list):
                papers_data = data
            else:
                papers_data = []
        
        # 限制加载数量
        if max_papers and max_papers > 0:
            papers_data = papers_data[:max_papers]
        
        papers = []
        for paper_data in papers_data:
            try:
                paper = self._parse_paper(paper_data)
                if paper:
                    papers.append(paper)
            except Exception as e:
                # 跳过无效的论文数据
                continue
        
        return papers
    
    def load_citations(
        self,
        path: Optional[str] = None,
        paper_ids: Optional[set] = None,
        max_citations: Optional[int] = None
    ) -> List[Citation]:
        """加载OpenCorpus格式的引用关系数据
        
        功能：
        - 支持JSON和JSONL格式
        - 验证source和target论文ID是否存在
        - 支持限制加载的引用数量
        
        Args:
            path: 引用数据文件路径。如果为None且self.data_dir存在，尝试使用默认路径
            paper_ids: 已加载的论文ID集合，用于验证source和target是否存在
            max_citations: 最大加载引用数量，用于快速测试。如果为None，加载所有引用
            
        Returns:
            引用对象列表
        """
        if path is None:
            if self.data_dir:
                # 尝试默认路径
                possible_paths = [
                    self.data_dir / "citations.json",
                    self.data_dir / "citations.jsonl",
                    self.data_dir / "opencorpus_citations.json",
                    self.data_dir / "opencorpus_citations.jsonl"
                ]
                for p in possible_paths:
                    if p.exists():
                        path = str(p)
                        break
                if path is None:
                    raise FileNotFoundError(f"未找到引用数据文件，请检查: {self.data_dir}")
            else:
                raise ValueError("必须提供引用数据文件路径")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"引用数据文件不存在: {path}")
        
        # 根据文件扩展名判断格式
        citations_data = []
        if path.suffix == ".jsonl":
            citations_data = load_jsonl(str(path))
        else:
            data = load_json(str(path))
            # 如果数据是字典，尝试获取citations键
            if isinstance(data, dict):
                citations_data = data.get("citations", data.get("data", data.get("items", [])))
            elif isinstance(data, list):
                citations_data = data
            else:
                citations_data = []
        
        # 限制加载数量
        if max_citations and max_citations > 0:
            citations_data = citations_data[:max_citations]
        
        citations = []
        for citation_data in citations_data:
            try:
                citation = self._parse_citation(citation_data, paper_ids)
                if citation:
                    citations.append(citation)
            except Exception as e:
                # 跳过无效的引用数据
                continue
        
        return citations
    
    def _parse_paper(self, paper_data: Dict[str, Any]) -> Optional[Paper]:
        """解析论文数据字典为Paper对象
        
        Args:
            paper_data: 论文数据字典
            
        Returns:
            Paper对象，如果数据无效则返回None
        """
        # OpenCorpus可能的字段名映射
        paper_id = paper_data.get("id") or paper_data.get("paper_id") or paper_data.get("_id")
        title = paper_data.get("title") or paper_data.get("paper_title") or ""
        abstract = paper_data.get("abstract") or paper_data.get("paper_abstract") or ""
        authors = paper_data.get("authors") or paper_data.get("author") or []
        year = paper_data.get("year") or paper_data.get("publication_year") or paper_data.get("pub_year")
        venue = paper_data.get("venue") or paper_data.get("journal") or paper_data.get("conference") or ""
        citation_count = paper_data.get("citation_count") or paper_data.get("num_citations") or paper_data.get("citations") or 0
        doi = paper_data.get("doi") or paper_data.get("DOI")
        
        # 验证必需字段
        if not paper_id:
            return None
        
        # 处理authors字段（可能是字符串列表或对象列表）
        if isinstance(authors, list):
            author_list = []
            for author in authors:
                if isinstance(author, str):
                    author_list.append(author)
                elif isinstance(author, dict):
                    author_name = author.get("name") or author.get("author") or ""
                    if author_name:
                        author_list.append(author_name)
            authors = author_list
        elif isinstance(authors, str):
            authors = [authors]
        else:
            authors = []
        
        # 处理year字段
        if year is None:
            year = 2000  # 默认年份
        elif isinstance(year, str):
            try:
                year = int(year)
            except:
                year = 2000
        
        # 处理citation_count字段
        if citation_count is None:
            citation_count = 0
        elif isinstance(citation_count, str):
            try:
                citation_count = int(citation_count)
            except:
                citation_count = 0
        
        return Paper(
            id=str(paper_id),
            title=str(title),
            abstract=str(abstract),
            authors=authors,
            year=int(year),
            venue=str(venue),
            citation_count=int(citation_count),
            doi=doi
        )
    
    def _parse_citation(
        self,
        citation_data: Dict[str, Any],
        paper_ids: Optional[set] = None
    ) -> Optional[Citation]:
        """解析引用数据字典为Citation对象
        
        Args:
            citation_data: 引用数据字典
            paper_ids: 已加载的论文ID集合，用于验证
            
        Returns:
            Citation对象，如果数据无效则返回None
        """
        # OpenCorpus可能的字段名映射
        source_id = citation_data.get("source") or citation_data.get("source_id") or citation_data.get("source_paper_id")
        target_id = citation_data.get("target") or citation_data.get("target_id") or citation_data.get("target_paper_id")
        context = citation_data.get("context") or citation_data.get("citation_context") or citation_data.get("text") or ""
        position = citation_data.get("position") or citation_data.get("citation_position") or "introduction"
        
        # 验证必需字段
        if not source_id or not target_id:
            return None
        
        # 验证论文ID是否存在
        if paper_ids is not None:
            if source_id not in paper_ids or target_id not in paper_ids:
                return None
        
        # 验证position字段
        allowed_positions = {"introduction", "methodology", "experiment", "discussion"}
        if position not in allowed_positions:
            position = "introduction"  # 默认位置
        
        return Citation(
            source_paper_id=str(source_id),
            target_paper_id=str(target_id),
            context=str(context),
            position=position
        )
    
    def load_dataset(
        self,
        papers_path: Optional[str] = None,
        citations_path: Optional[str] = None,
        max_papers: Optional[int] = None,
        max_citations: Optional[int] = None
    ) -> Tuple[List[Paper], List[Citation]]:
        """加载完整的OpenCorpus数据集
        
        Args:
            papers_path: 论文数据文件路径
            citations_path: 引用数据文件路径
            max_papers: 最大加载论文数量
            max_citations: 最大加载引用数量
            
        Returns:
            (论文列表, 引用列表) 元组
        """
        # 先加载论文
        papers = self.load_papers(papers_path, max_papers=max_papers)
        paper_ids = {paper.id for paper in papers}
        
        # 再加载引用（使用论文ID集合进行验证）
        citations = self.load_citations(
            citations_path,
            paper_ids=paper_ids,
            max_citations=max_citations
        )
        
        return papers, citations

