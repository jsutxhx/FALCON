"""引用功能适应性评估指标模块"""
from typing import List, Dict, Set, Optional
from collections import defaultdict
import numpy as np
from src.evaluation.accuracy_metrics import AccuracyMetrics


class FunctionAdaptabilityMetrics:
    """引用功能适应性评估指标
    
    功能：
    - 计算推荐系统对不同引用功能的适应能力
    - 包括：Function Match Accuracy (FMA), Function-specific Ranking Quality
    
    所有方法都是静态方法，可以直接调用
    """
    
    FUNCTIONS = ["background", "use", "compare", "inspire"]
    
    @staticmethod
    def function_match_accuracy(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """计算引用功能匹配准确率（Function Match Accuracy, FMA）
        
        功能：
        - 计算各功能类别的匹配准确率
        - 公式: FMA_f = |{correct predictions for function f}| / |{samples with function f}|
        
        Args:
            predictions: 预测结果列表，每个元素是一个字典，包含：
                - "recommendations": 推荐列表，每个推荐包含"paper_id"
                - "citation_function": 预测的引用功能（可选）
            ground_truth: 真实标签列表，每个元素是一个字典，包含：
                - "ground_truth": 真实相关论文ID集合
                - "citation_function": 真实的引用功能
        
        Returns:
            包含各功能FMA值的字典，键为功能名称，值为FMA值
            包含"overall"键表示总体FMA
        """
        if not predictions or not ground_truth:
            return {}
        
        if len(predictions) != len(ground_truth):
            return {}
        
        # 按功能分组统计
        function_correct = defaultdict(int)
        function_total = defaultdict(int)
        
        for pred, gt in zip(predictions, ground_truth):
            # 获取真实引用功能
            gt_function = gt.get("citation_function", "unknown")
            if gt_function not in FunctionAdaptabilityMetrics.FUNCTIONS:
                continue
            
            # 获取预测和真实的论文ID集合
            pred_papers = set()
            if "recommendations" in pred:
                for rec in pred["recommendations"]:
                    if isinstance(rec, dict):
                        paper_id = rec.get("paper_id") or rec.get("paperId")
                    elif hasattr(rec, "paper_id"):
                        paper_id = rec.paper_id
                    else:
                        paper_id = str(rec)
                    if paper_id:
                        pred_papers.add(str(paper_id))
            
            gt_papers = set()
            if "ground_truth" in gt:
                gt_data = gt["ground_truth"]
                if isinstance(gt_data, (list, set)):
                    gt_papers = {str(pid) for pid in gt_data}
                elif isinstance(gt_data, str):
                    gt_papers = {gt_data}
            
            # 只计算有ground_truth的查询（避免无ground_truth的查询拉低FMA）
            if not gt_papers:
                continue
            
            # 统计该功能的样本数
            function_total[gt_function] += 1
            
            # 检查是否有正确的推荐（至少有一个推荐在真实标签中）
            if pred_papers & gt_papers:
                function_correct[gt_function] += 1
        
        # 计算每个功能的FMA
        result = {}
        for func in FunctionAdaptabilityMetrics.FUNCTIONS:
            total = function_total.get(func, 0)
            correct = function_correct.get(func, 0)
            result[func] = correct / total if total > 0 else 0.0
        
        # 计算总体FMA
        total_samples = sum(function_total.values())
        total_correct = sum(function_correct.values())
        result["overall"] = total_correct / total_samples if total_samples > 0 else 0.0
        
        return result
    
    @staticmethod
    def function_ranking_quality(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict[str, float]:
        """计算功能特定的排序质量（Function-specific Ranking Quality）
        
        功能：
        - 计算各功能类别的MAP值
        - 用于评估不同引用功能下的推荐排序质量
        
        Args:
            predictions: 预测结果列表，每个元素是一个字典，包含：
                - "recommendations": 推荐列表，每个推荐包含"paper_id"（按得分降序排列）
            ground_truth: 真实标签列表，每个元素是一个字典，包含：
                - "ground_truth": 真实相关论文ID集合
                - "citation_function": 真实的引用功能
        
        Returns:
            包含各功能MAP值的字典，键为功能名称，值为MAP值
        """
        if not predictions or not ground_truth:
            return {}
        
        if len(predictions) != len(ground_truth):
            return {}
        
        # 按功能分组
        function_results = defaultdict(lambda: {"recommended": [], "ground_truth": []})
        
        for pred, gt in zip(predictions, ground_truth):
            # 获取真实引用功能
            gt_function = gt.get("citation_function", "unknown")
            if gt_function not in FunctionAdaptabilityMetrics.FUNCTIONS:
                continue
            
            # 提取推荐论文ID列表（按顺序）
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
            
            function_results[gt_function]["recommended"].append(recommended)
            function_results[gt_function]["ground_truth"].append(gt_papers)
        
        # 计算每个功能的MAP
        result = {}
        for func in FunctionAdaptabilityMetrics.FUNCTIONS:
            data = function_results.get(func)
            if data and data["recommended"] and data["ground_truth"]:
                result[func] = AccuracyMetrics.mean_average_precision(
                    data["recommended"],
                    data["ground_truth"]
                )
            else:
                result[func] = 0.0
        
        return result


