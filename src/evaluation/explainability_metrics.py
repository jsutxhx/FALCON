"""可解释性评估指标模块"""
from typing import List, Set, Union, Dict, Any
import networkx as nx


class ExplainabilityMetrics:
    """可解释性评估指标
    
    功能：
    - 计算推荐系统的可解释性指标
    - 包括：Path Coverage, Evidence Verifiability
    
    所有方法都是静态方法，可以直接调用
    """
    
    @staticmethod
    def path_coverage(
        recommendations: List,
        graph: Union[nx.Graph, nx.DiGraph, Any],
        query_paper_id: str = None
    ) -> float:
        """计算路径覆盖率（Path Coverage）
        
        功能：
        - 计算推荐中有推理路径的推荐比例
        - 如果推荐中的论文在知识图谱中有路径连接（到查询论文或到其他推荐论文），则认为有推理路径
        - 公式: Path Coverage = |Recommendations with Path| / |Total Recommendations|
        
        Args:
            recommendations: 推荐列表，每个元素应该有paper_id属性（Recommendation对象或字典）
            graph: 知识图谱对象，可以是networkx.Graph/DiGraph或KnowledgeGraph对象
            query_paper_id: 查询论文ID（可选），如果提供，会检查推荐论文到查询论文的路径
            
        Returns:
            路径覆盖率，范围在[0, 1]之间
            如果recommendations为空，返回0.0
        """
        if not recommendations:
            return 0.0
        
        # 获取networkx图对象
        nx_graph = ExplainabilityMetrics._get_nx_graph(graph)
        if nx_graph is None:
            return 0.0
        
        # 提取推荐论文ID列表
        recommended_paper_ids = []
        for rec in recommendations:
            if isinstance(rec, dict):
                paper_id = rec.get("paper_id") or rec.get("paperId")
            elif hasattr(rec, 'paper_id'):
                paper_id = rec.paper_id
            else:
                continue
            
            if paper_id:
                recommended_paper_ids.append(paper_id)
        
        if not recommended_paper_ids:
            return 0.0
        
        # 统计有路径的推荐数量
        recommendations_with_path = 0
        
        for paper_id in recommended_paper_ids:
            has_path = False
            
            # 如果提供了查询论文ID，检查到查询论文的路径
            if query_paper_id and query_paper_id in nx_graph:
                if paper_id in nx_graph:
                    # 检查是否有路径（双向检查，因为可能是有向图）
                    if nx.has_path(nx_graph, query_paper_id, paper_id) or \
                       nx.has_path(nx_graph, paper_id, query_paper_id):
                        has_path = True
            
            # 检查到其他推荐论文的路径（表示推荐之间有连接）
            if not has_path and paper_id in nx_graph:
                for other_paper_id in recommended_paper_ids:
                    if other_paper_id != paper_id and other_paper_id in nx_graph:
                        if nx.has_path(nx_graph, paper_id, other_paper_id) or \
                           nx.has_path(nx_graph, other_paper_id, paper_id):
                            has_path = True
                            break
            
            # 检查论文节点是否在图中（至少存在节点也算有路径）
            if not has_path and paper_id in nx_graph:
                # 如果节点存在且有邻居，认为有路径
                if len(list(nx_graph.neighbors(paper_id))) > 0:
                    has_path = True
            
            if has_path:
                recommendations_with_path += 1
        
        # 计算覆盖率
        coverage = recommendations_with_path / len(recommended_paper_ids)
        
        return float(coverage)
    
    @staticmethod
    def evidence_verifiability(
        recommendations: List,
        paper_db: Union[Dict[str, Any], Any]
    ) -> float:
        """计算证据可验证性（Evidence Verifiability）
        
        功能：
        - 计算推荐中可验证的推荐比例
        - 可验证的条件：
          1. 推荐的论文ID存在于paper_db中
          2. 推荐理由（reason）不为空
        - 公式: Evidence Verifiability = |Verifiable Recommendations| / |Total Recommendations|
        
        Args:
            recommendations: 推荐列表，每个元素应该有paper_id和reason属性（Recommendation对象或字典）
            paper_db: 论文数据库，可以是Dict[str, Paper]或具有get_paper(id)方法的对象
                     或具有.papers属性（Dict[str, Paper]）的对象，或支持索引访问的对象
            
        Returns:
            证据可验证性，范围在[0, 1]之间
            如果recommendations为空，返回0.0
        """
        if not recommendations:
            return 0.0
        
        verifiable_count = 0
        
        for rec in recommendations:
            # 提取paper_id
            if isinstance(rec, dict):
                paper_id = rec.get("paper_id") or rec.get("paperId")
                reason = rec.get("reason") or rec.get("explanation")
            elif hasattr(rec, 'paper_id'):
                paper_id = rec.paper_id
                reason = getattr(rec, 'reason', None) or getattr(rec, 'explanation', None)
            else:
                continue
            
            # 检查paper_id是否存在
            paper_exists = ExplainabilityMetrics._paper_exists(paper_id, paper_db)
            
            # 检查reason是否不为空
            has_reason = bool(reason and str(reason).strip())
            
            # 如果paper_id存在且有reason，则认为可验证
            if paper_exists and has_reason:
                verifiable_count += 1
        
        # 计算可验证性
        verifiability = verifiable_count / len(recommendations)
        
        return float(verifiability)
    
    @staticmethod
    def _get_nx_graph(graph: Any) -> Union[nx.Graph, nx.DiGraph, None]:
        """从各种图对象中提取networkx图对象
        
        Args:
            graph: 图对象，可能是networkx.Graph/DiGraph或KnowledgeGraph对象
            
        Returns:
            networkx图对象，如果无法提取则返回None
        """
        if isinstance(graph, (nx.Graph, nx.DiGraph)):
            return graph
        elif hasattr(graph, 'graph') and isinstance(graph.graph, (nx.Graph, nx.DiGraph)):
            # KnowledgeGraph对象
            return graph.graph
        elif hasattr(graph, 'nx_graph'):
            return graph.nx_graph
        else:
            return None
    
    @staticmethod
    def _paper_exists(paper_id: str, paper_db: Any) -> bool:
        """检查论文ID是否存在于论文数据库中
        
        Args:
            paper_id: 论文ID
            paper_db: 论文数据库，可以是Dict[str, Paper]或具有get_paper(id)方法的对象
                     或具有.papers属性（Dict[str, Paper]）的对象，或支持索引访问的对象
        
        Returns:
            如果论文ID存在则返回True，否则返回False
        """
        if not paper_id:
            return False
        
        if paper_db is None:
            return False
        
        # 如果是字典
        if isinstance(paper_db, dict):
            return paper_id in paper_db
        
        # 如果有get_paper方法
        if hasattr(paper_db, 'get_paper'):
            try:
                paper = paper_db.get_paper(paper_id)
                return paper is not None
            except:
                return False
        
        # 如果有papers属性
        if hasattr(paper_db, 'papers') and isinstance(paper_db.papers, dict):
            return paper_id in paper_db.papers
        
        # 如果支持索引访问
        try:
            paper = paper_db[paper_id]
            return paper is not None
        except (KeyError, TypeError):
            return False


