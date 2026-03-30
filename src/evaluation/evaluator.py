"""评估器主类模块"""
from typing import List, Set, Dict, Any, Union, Optional, Callable
from src.evaluation.accuracy_metrics import AccuracyMetrics
from src.evaluation.diversity_metrics import DiversityMetrics
from src.evaluation.explainability_metrics import ExplainabilityMetrics
from src.evaluation.function_metrics import FunctionAdaptabilityMetrics
from src.data_processing.data_structures import Paper, Recommendation
import networkx as nx
import numpy as np


class Evaluator:
    """评估器主类
    
    功能：
    - 整合所有评估指标
    - 提供统一的评估接口
    - 支持准确性、多样性、可解释性指标的评估
    
    Attributes:
        k_values: 用于计算P@K和R@K的K值列表，默认为[1, 3, 5, 10]
        similarity_fn: 用于计算ILD的相似度函数（可选）
        all_topics: 用于计算Topic Coverage的所有主题集合（可选）
        graph: 知识图谱对象，用于计算Path Coverage（可选）
        paper_db: 论文数据库，用于计算Evidence Verifiability（可选）
    """
    
    def __init__(
        self,
        k_values: List[int] = None,
        similarity_fn: Optional[Callable] = None,
        all_topics: Optional[Set[str]] = None,
        graph: Optional[Any] = None,
        paper_db: Optional[Any] = None
    ):
        """初始化评估器
        
        Args:
            k_values: 用于计算P@K和R@K的K值列表，默认为[5, 10, 20]（根据论文实验设置）
            similarity_fn: 用于计算ILD的相似度函数，接受两个Paper对象，返回相似度值
            all_topics: 用于计算Topic Coverage的所有主题集合
            graph: 知识图谱对象，用于计算Path Coverage
            paper_db: 论文数据库，用于计算Evidence Verifiability
        """
        # 根据论文实验设置，默认使用[5, 10, 20]
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        self.similarity_fn = similarity_fn
        self.all_topics = all_topics
        self.graph = graph
        self.paper_db = paper_db
    
    def evaluate(
        self,
        predictions: Union[List[Recommendation], List[List[str]], List[Paper]],
        ground_truth: Union[Set[str], List[Set[str]]],
        query_paper_id: Optional[str] = None,
        papers: Optional[List[Paper]] = None
    ) -> Dict[str, float]:
        """执行完整评估
        
        功能：
        - 整合所有评估指标
        - 计算准确性、多样性、可解释性指标
        - 返回包含所有指标的字典
        
        Args:
            predictions: 预测结果，可以是：
                - List[Recommendation]: 推荐对象列表
                - List[List[str]]: 多个查询的排序结果列表（每个元素是论文ID列表）
                - List[Paper]: 论文对象列表（用于多样性指标）
            ground_truth: 真实标签，可以是：
                - Set[str]: 单个查询的真实标签集合
                - List[Set[str]]: 多个查询的真实标签列表
            query_paper_id: 查询论文ID（可选），用于Path Coverage计算
            papers: 论文对象列表（可选），如果predictions是Recommendation或ID列表，需要提供此参数用于多样性指标
            
        Returns:
            包含所有评估指标的字典，键为指标名称，值为指标值
        """
        results = {}
        
        # 1. 准确性指标
        accuracy_results = self._evaluate_accuracy(predictions, ground_truth)
        results.update(accuracy_results)
        
        # 2. 多样性指标
        if self.similarity_fn is not None or self.all_topics is not None:
            diversity_results = self._evaluate_diversity(predictions, papers)
            results.update(diversity_results)
        
        # 3. 可解释性指标
        if self.graph is not None or self.paper_db is not None:
            explainability_results = self._evaluate_explainability(
                predictions, query_paper_id
            )
            results.update(explainability_results)
        
        return results
    
    def evaluate_with_functions(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, Any]:
        """执行包含功能适应性指标的完整评估
        
        功能：
        - 整合所有评估指标，包括功能适应性指标
        - 计算准确性、多样性、可解释性、功能适应性指标
        - 返回包含所有指标的字典
        
        Args:
            predictions: 预测结果列表，每个元素是一个字典，包含：
                - "recommendations": 推荐列表
                - "citation_function": 预测的引用功能（可选）
            ground_truth: 真实标签列表，每个元素是一个字典，包含：
                - "ground_truth": 真实相关论文ID集合
                - "citation_function": 真实的引用功能
        
        Returns:
            包含所有评估指标的字典，包括：
            - 准确性指标：P@5, P@10, P@20, R@5, R@10, R@20, MAP, MRR
            - 多样性指标：ILD, Topic_Coverage, Temporal_Diversity
            - 可解释性指标：Path_Coverage, Evidence_Verifiability
            - 功能适应性指标：FMA_background, FMA_use, FMA_compare, FMA_inspire, FMA_overall
        """
        results = {}
        
        if not predictions or not ground_truth:
            return results
        
        # 1. 准确性指标（从predictions和ground_truth中提取）
        accuracy_results = self._evaluate_accuracy_from_dicts(predictions, ground_truth)
        results.update(accuracy_results)
        
        # 2. 多样性指标（需要papers信息）
        if self.similarity_fn is not None or self.all_topics is not None:
            diversity_results = self._evaluate_diversity_from_dicts(predictions)
            results.update(diversity_results)
        
        # 3. 可解释性指标
        if self.graph is not None or self.paper_db is not None:
            explainability_results = self._evaluate_explainability_from_dicts(predictions)
            results.update(explainability_results)
        
        # 4. 功能适应性指标
        fma_results = FunctionAdaptabilityMetrics.function_match_accuracy(
            predictions, ground_truth
        )
        for func, value in fma_results.items():
            results[f"FMA_{func}"] = value
        
        return results
    
    def _evaluate_accuracy(
        self,
        predictions: Union[List[Recommendation], List[List[str]], List[Paper]],
        ground_truth: Union[Set[str], List[Set[str]]]
    ) -> Dict[str, float]:
        """评估准确性指标"""
        results = {}
        
        # 处理空预测
        if not predictions:
            return results
        
        # 提取论文ID列表
        if isinstance(predictions[0], Recommendation):
            # 如果是Recommendation对象列表，提取paper_id
            recommended_ids = [rec.paper_id for rec in predictions]
        elif isinstance(predictions[0], Paper):
            # 如果是Paper对象列表，提取id
            recommended_ids = [paper.id for paper in predictions]
        elif isinstance(predictions[0], list):
            # 如果是多个查询的排序结果列表
            # 计算MAP和MRR
            if isinstance(ground_truth, list) and len(ground_truth) == len(predictions):
                map_value = AccuracyMetrics.mean_average_precision(predictions, ground_truth)
                mrr_value = AccuracyMetrics.mean_reciprocal_rank(predictions, ground_truth)
                results["MAP"] = map_value
                results["MRR"] = mrr_value
            return results
        else:
            # 假设是字符串列表（论文ID列表）
            recommended_ids = list(predictions)
        
        # 处理ground_truth
        if isinstance(ground_truth, set):
            gt_set = ground_truth
        elif isinstance(ground_truth, list) and len(ground_truth) > 0:
            # 如果是列表，取第一个（单个查询的情况）
            gt_set = ground_truth[0] if isinstance(ground_truth[0], set) else set(ground_truth[0])
        else:
            gt_set = set()
        
        # 计算P@K和R@K
        for k in self.k_values:
            p_at_k = AccuracyMetrics.precision_at_k(recommended_ids, gt_set, k)
            r_at_k = AccuracyMetrics.recall_at_k(recommended_ids, gt_set, k)
            results[f"P@{k}"] = p_at_k
            results[f"R@{k}"] = r_at_k
        
        return results
    
    def _evaluate_diversity(
        self,
        predictions: Union[List[Recommendation], List[List[str]], List[Paper]],
        papers: Optional[List[Paper]] = None
    ) -> Dict[str, float]:
        """评估多样性指标"""
        results = {}
        
        # 处理空预测
        if not predictions:
            return results
        
        # 获取Paper对象列表
        paper_list = None
        
        if isinstance(predictions[0], Paper):
            paper_list = predictions
        elif isinstance(predictions[0], Recommendation):
            # 从Recommendation对象中提取paper_id，然后从papers中查找
            if papers is not None:
                paper_dict = {paper.id: paper for paper in papers}
                paper_list = [
                    paper_dict.get(rec.paper_id) 
                    for rec in predictions 
                    if rec.paper_id in paper_dict
                ]
                paper_list = [p for p in paper_list if p is not None]
        elif isinstance(predictions[0], str):
            # 如果是论文ID列表，从papers中查找
            if papers is not None:
                paper_dict = {paper.id: paper for paper in papers}
                paper_list = [
                    paper_dict.get(paper_id) 
                    for paper_id in predictions 
                    if paper_id in paper_dict
                ]
                paper_list = [p for p in paper_list if p is not None]
        
        if paper_list is None or len(paper_list) < 2:
            return results
        
        # 计算ILD
        if self.similarity_fn is not None:
            ild = DiversityMetrics.intra_list_distance(paper_list, self.similarity_fn)
            results["ILD"] = ild
        
        # 计算Topic Coverage
        if self.all_topics is not None:
            topic_coverage = DiversityMetrics.topic_coverage(paper_list, self.all_topics)
            results["Topic_Coverage"] = topic_coverage
        
        return results
    
    def _evaluate_explainability(
        self,
        predictions: Union[List[Recommendation], List[List[str]], List[Paper]],
        query_paper_id: Optional[str] = None
    ) -> Dict[str, float]:
        """评估可解释性指标"""
        results = {}
        
        # 处理空预测
        if not predictions:
            return results
        
        # 转换为Recommendation对象列表或提取推荐信息
        recommendations = []
        
        if isinstance(predictions[0], Recommendation):
            recommendations = predictions
        elif isinstance(predictions[0], Paper):
            # 将Paper对象转换为字典格式
            recommendations = [
                {"paper_id": paper.id, "reason": f"Recommended paper: {paper.title}"}
                for paper in predictions
            ]
        elif isinstance(predictions[0], str):
            # 如果是论文ID列表，转换为字典格式
            recommendations = [
                {"paper_id": paper_id, "reason": f"Recommended paper ID: {paper_id}"}
                for paper_id in predictions
            ]
        
        if not recommendations:
            return results
        
        # 计算Path Coverage
        if self.graph is not None:
            path_coverage = ExplainabilityMetrics.path_coverage(
                recommendations, self.graph, query_paper_id
            )
            results["Path_Coverage"] = path_coverage
        
        # 计算Evidence Verifiability
        if self.paper_db is not None:
            evidence_verifiability = ExplainabilityMetrics.evidence_verifiability(
                recommendations, self.paper_db
            )
            results["Evidence_Verifiability"] = evidence_verifiability
        
        return results
    
    def _evaluate_accuracy_from_dicts(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """从字典格式的预测和真实标签中评估准确性指标"""
        results = {}
        
        if not predictions or not ground_truth:
            return results
        
        # 提取推荐列表和真实标签列表
        all_recommended = []
        all_ground_truth = []
        
        # 调试：统计ground_truth分布（在提取时）
        gt_counts_before = []
        
        for pred, gt in zip(predictions, ground_truth):
            # 提取推荐论文ID列表
            recommended = []
            if "recommendations" in pred:
                for rec in pred["recommendations"]:
                    if isinstance(rec, dict):
                        paper_id = rec.get("paper_id") or rec.get("paperId")
                    elif hasattr(rec, "paper_id"):
                        paper_id = rec.paper_id
                    else:
                        paper_id = str(rec)
                    if paper_id:
                        recommended.append(str(paper_id))
            
            # 提取真实标签集合
            gt_papers = set()
            if "ground_truth" in gt:
                gt_data = gt["ground_truth"]
                if isinstance(gt_data, (list, set)):
                    gt_papers = {str(pid) for pid in gt_data}
                elif isinstance(gt_data, str):
                    gt_papers = {gt_data}
            
            all_recommended.append(recommended)
            all_ground_truth.append(gt_papers)
            gt_counts_before.append(len(gt_papers))
        
        # 调试：统计提取后的ground_truth分布
        if gt_counts_before:
            import numpy as np
            gt_counts_array = np.array(gt_counts_before)
            results['_debug_gt_distribution'] = {
                'total': len(gt_counts_before),
                'with_gt': int(np.sum(gt_counts_array > 0)),
                'gt_le10': int(np.sum(gt_counts_array <= 10)),
                'gt_le5': int(np.sum(gt_counts_array <= 5)),
                'gt_le20': int(np.sum(gt_counts_array <= 20)),
                'min': int(np.min(gt_counts_array)),
                'max': int(np.max(gt_counts_array)),
                'mean': float(np.mean(gt_counts_array)),
                'median': float(np.median(gt_counts_array))
            }
        
            # 计算MAP和MRR
            if all_recommended and all_ground_truth:
                map_value = AccuracyMetrics.mean_average_precision(all_recommended, all_ground_truth)
                mrr_value = AccuracyMetrics.mean_reciprocal_rank(all_recommended, all_ground_truth)
                results["MAP"] = map_value
                results["MRR"] = mrr_value
                
                # 先计算统计信息（在循环外，只计算一次）
                queries_with_gt = sum(1 for gt in all_ground_truth if gt)
                queries_with_gt_le10 = sum(1 for gt in all_ground_truth if gt and len(gt) <= 10)
                queries_with_gt_le10_and_rec = sum(1 for rec, gt in zip(all_recommended, all_ground_truth) 
                                                   if gt and len(gt) <= 10 and rec)
                
                # 计算每个查询的P@K和R@K，然后取平均
                # 只对有ground_truth的查询计算指标（避免无ground_truth的查询拉低平均值）
                for k in self.k_values:
                    p_at_k_list = []
                    r_at_k_list = []
                    r_at_k_details = []  # 用于调试
                    
                    for idx, (rec, gt) in enumerate(zip(all_recommended, all_ground_truth)):
                        # 跳过没有ground_truth的查询
                        if not gt:
                            continue
                        p_at_k = AccuracyMetrics.precision_at_k(rec, gt, k)
                        r_at_k = AccuracyMetrics.recall_at_k(rec, gt, k)
                        p_at_k_list.append(p_at_k)
                        r_at_k_list.append(r_at_k)
                        
                        # 记录详细信息用于调试（只记录前10个）
                        if len(r_at_k_details) < 10:
                            intersection = len(set(rec[:k]) & gt)
                            r_at_k_details.append({
                                'query_idx': idx,
                                'gt_count': len(gt),
                                'rec_count': len(rec),
                                'top_k_count': min(k, len(rec)),
                                'intersection': intersection,
                                'r_at_k': r_at_k,
                                'p_at_k': p_at_k
                            })
                    
                    if p_at_k_list:
                        results[f"P@{k}"] = np.mean(p_at_k_list)
                    if r_at_k_list:
                        results[f"R@{k}"] = np.mean(r_at_k_list)
                        # 添加调试信息
                        if r_at_k_list:
                            results[f'R@{k}_debug'] = {
                                'count': len(r_at_k_list),
                                'min': float(np.min(r_at_k_list)),
                                'max': float(np.max(r_at_k_list)),
                                'mean': float(np.mean(r_at_k_list)),
                                'std': float(np.std(r_at_k_list)),
                                'details': r_at_k_details
                            }
                
                # 添加统计信息（在循环外，只计算一次）
                results['_debug_stats'] = {
                    'total_queries': len(all_recommended),
                    'queries_with_gt': queries_with_gt,
                    'queries_with_gt_le10': queries_with_gt_le10,
                    'queries_with_gt_le10_and_rec': queries_with_gt_le10_and_rec,
                    'r_at_k_filtered_count': queries_with_gt  # 实际参与R@K计算的查询数（所有有ground_truth的查询）
                }
        
        return results
    
    def _evaluate_diversity_from_dicts(
        self,
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """从字典格式的预测中评估多样性指标"""
        results = {}
        
        if not predictions:
            return results
        
        # 需要从paper_db中获取Paper对象
        if self.paper_db is None:
            return results
        
        # 收集所有推荐的论文
        all_papers = []
        for pred in predictions:
            if "recommendations" in pred:
                for rec in pred["recommendations"]:
                    paper_id = None
                    if isinstance(rec, dict):
                        paper_id = rec.get("paper_id") or rec.get("paperId")
                    elif hasattr(rec, "paper_id"):
                        paper_id = rec.paper_id
                    else:
                        paper_id = str(rec)
                    
                    if paper_id:
                        # 从paper_db中获取Paper对象
                        paper = None
                        if isinstance(self.paper_db, dict):
                            paper = self.paper_db.get(paper_id)
                        elif hasattr(self.paper_db, "get"):
                            paper = self.paper_db.get(paper_id)
                        
                        if paper and isinstance(paper, Paper):
                            all_papers.append(paper)
        
        if not all_papers:
            return results
        
        # 计算ILD（如果论文数量太多，采样计算以提高速度）
        if self.similarity_fn is not None and len(all_papers) >= 2:
            # 如果论文数量超过1000，采样计算以提高速度
            if len(all_papers) > 1000:
                import random
                random.seed(42)
                sampled_papers = random.sample(all_papers, min(1000, len(all_papers)))
                ild = DiversityMetrics.intra_list_distance(sampled_papers, self.similarity_fn)
            else:
                ild = DiversityMetrics.intra_list_distance(all_papers, self.similarity_fn)
            results["ILD"] = ild
        
        # 计算Topic Coverage
        if self.all_topics is not None:
            topic_coverage = DiversityMetrics.topic_coverage(all_papers, self.all_topics)
            results["Topic_Coverage"] = topic_coverage
        
        # 计算Temporal Diversity
        temporal_diversity = DiversityMetrics.temporal_diversity(all_papers)
        results["Temporal_Diversity"] = temporal_diversity
        
        return results
    
    def _evaluate_explainability_from_dicts(
        self,
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """从字典格式的预测中评估可解释性指标"""
        results = {}
        
        if not predictions:
            return results
        
        # 收集所有推荐
        all_recommendations = []
        for pred in predictions:
            if "recommendations" in pred:
                for rec in pred["recommendations"]:
                    if isinstance(rec, dict):
                        all_recommendations.append(rec)
                    elif hasattr(rec, "paper_id"):
                        # 转换为字典
                        all_recommendations.append({
                            "paper_id": rec.paper_id,
                            "reason": getattr(rec, "reason", "")
                        })
        
        if not all_recommendations:
            return results
        
        # 计算Path Coverage
        if self.graph is not None:
            path_coverage = ExplainabilityMetrics.path_coverage(
                all_recommendations, self.graph
            )
            results["Path_Coverage"] = path_coverage
        
        # 计算Evidence Verifiability
        if self.paper_db is not None:
            evidence_verifiability = ExplainabilityMetrics.evidence_verifiability(
                all_recommendations, self.paper_db
            )
            results["Evidence_Verifiability"] = evidence_verifiability
        
        return results

