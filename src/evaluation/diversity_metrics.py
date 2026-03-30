"""多样性评估指标模块"""
from typing import List, Set, Callable
import numpy as np


class DiversityMetrics:
    """多样性评估指标
    
    功能：
    - 计算推荐系统的多样性指标
    - 包括：Intra-List Distance (ILD), Topic Coverage
    
    所有方法都是静态方法，可以直接调用
    """
    
    @staticmethod
    def intra_list_distance(
        papers: List,
        similarity_fn: Callable
    ) -> float:
        """计算列表内距离（Intra-List Distance, ILD）
        
        功能：
        - 计算推荐列表中论文之间的平均距离
        - 公式: ILD = (2 / K(K-1)) Σ_i Σ_{j>i} (1 - sim(p_i, p_j))
        其中：
        - K: 论文数量
        - sim(p_i, p_j): 论文i和论文j之间的相似度（范围[0, 1]）
        - ILD值越高，表示列表越多样化
        
        Args:
            papers: 论文列表
            similarity_fn: 相似度计算函数，接受两个Paper对象，返回相似度值（范围[0, 1]）
            
        Returns:
            ILD值，范围在[0, 1]之间
            如果论文数量少于2，返回0.0
        """
        k = len(papers)
        if k < 2:
            return 0.0
        
        if k == 2:
            # 只有两个论文，直接计算
            sim = similarity_fn(papers[0], papers[1])
            return 1.0 - sim
        
        # 计算所有论文对之间的距离
        total_distance = 0.0
        count = 0
        
        for i in range(k):
            for j in range(i + 1, k):
                sim = similarity_fn(papers[i], papers[j])
                # 距离 = 1 - 相似度
                distance = 1.0 - sim
                total_distance += distance
                count += 1
        
        # 平均距离
        if count == 0:
            return 0.0
        
        ild = total_distance / count
        
        return float(ild)
    
    @staticmethod
    def topic_coverage(
        papers: List,
        all_topics: Set[str]
    ) -> float:
        """计算主题覆盖率（Topic Coverage）
        
        功能：
        - 计算推荐列表中覆盖的主题数量占所有主题的比例
        - 公式: Topic Coverage = |Covered Topics| / |All Topics|
        
        Args:
            papers: 论文列表，每个Paper对象应该有topics属性（List[str]或Set[str]）
            all_topics: 所有可能的主题集合
            
        Returns:
            主题覆盖率，范围在[0, 1]之间
            如果all_topics为空，返回0.0
            如果papers为空，返回0.0
        """
        if not all_topics:
            return 0.0
        
        if not papers:
            return 0.0
        
        # 收集所有被覆盖的主题（只计算在all_topics中的主题）
        covered_topics = set()
        
        for paper in papers:
            # 尝试从paper对象获取topics属性
            # 支持多种格式：List[str], Set[str], 或通过方法获取
            paper_topics = set()
            
            if hasattr(paper, 'topics'):
                topics = paper.topics
                if isinstance(topics, (list, tuple)):
                    paper_topics.update(topics)
                elif isinstance(topics, set):
                    paper_topics.update(topics)
                elif isinstance(topics, str):
                    # 如果是单个字符串，将其作为主题
                    paper_topics.add(topics)
            elif hasattr(paper, 'get_topics'):
                # 如果有get_topics方法，调用它
                topics = paper.get_topics()
                if isinstance(topics, (list, tuple, set)):
                    if isinstance(topics, set):
                        paper_topics.update(topics)
                    else:
                        paper_topics.update(topics)
                elif isinstance(topics, str):
                    paper_topics.add(topics)
            else:
                # 如果没有topics属性，从标题和摘要中提取关键词
                # 这与评估代码中的提取逻辑一致
                import re
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
                
                text = ""
                if hasattr(paper, 'title') and paper.title:
                    text += " " + paper.title
                if hasattr(paper, 'abstract') and paper.abstract:
                    text += " " + paper.abstract
                
                # 提取单词（只保留字母，长度>=4）
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                # 过滤停用词
                keywords = [w for w in words if w not in stop_words]
                paper_topics = set(keywords)
            
            # 只添加在all_topics中的主题
            covered_topics.update(paper_topics & all_topics)
        
        # 计算覆盖率
        coverage = len(covered_topics) / len(all_topics)
        
        return float(coverage)
    
    @staticmethod
    def temporal_diversity(papers: List) -> float:
        """计算时间多样性（Temporal Diversity）
        
        功能：
        - 计算推荐论文年份的标准差
        - 公式: TD = std(years)
        - 值越高表示时间跨度越大，推荐越多样化
        
        Args:
            papers: 论文列表，每个Paper对象应该有year属性（整数）
            
        Returns:
            时间多样性值（年份标准差）
            如果论文数量少于2或没有年份信息，返回0.0
        """
        if not papers or len(papers) < 2:
            return 0.0
        
        # 提取年份
        years = []
        for paper in papers:
            if hasattr(paper, 'year') and paper.year:
                try:
                    year = int(paper.year)
                    if year > 1900 and year < 2100:  # 合理的年份范围
                        years.append(year)
                except (ValueError, TypeError):
                    continue
        
        if len(years) < 2:
            return 0.0
        
        # 计算标准差
        std_dev = np.std(years)
        
        return float(std_dev)

