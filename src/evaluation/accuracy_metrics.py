"""准确性评估指标模块"""
from typing import List, Set
import numpy as np


class AccuracyMetrics:
    """准确性评估指标
    
    功能：
    - 计算推荐系统的准确性指标
    - 包括：Precision@K, Recall@K, MAP, MRR
    
    所有方法都是静态方法，可以直接调用
    """
    
    @staticmethod
    def precision_at_k(
        recommended: List[str],
        ground_truth: Set[str],
        k: int
    ) -> float:
        """计算Precision@K
        
        功能：
        - 计算前K个推荐中正确推荐的比例
        - 公式: P@K = |Recommended_K ∩ GroundTruth| / K
        
        Args:
            recommended: 推荐列表（论文ID列表），按推荐顺序排列
            ground_truth: 真实标签集合（论文ID集合）
            k: 计算前K个推荐
            
        Returns:
            Precision@K值，范围在[0, 1]之间
            如果k <= 0或recommended为空，返回0.0
        """
        if k <= 0:
            return 0.0
        
        if not recommended:
            return 0.0
        
        # 取前k个推荐
        top_k = recommended[:k]
        
        # 计算交集：前k个推荐中属于真实标签的数量
        intersection = len(set(top_k) & ground_truth)
        
        # P@K = 交集大小 / k
        precision = intersection / k
        
        return float(precision)
    
    @staticmethod
    def recall_at_k(
        recommended: List[str],
        ground_truth: Set[str],
        k: int
    ) -> float:
        """计算Recall@K
        
        功能：
        - 计算前K个推荐中覆盖的真实标签比例
        - 公式: R@K = |Recommended_K ∩ GroundTruth| / |GroundTruth|
        
        Args:
            recommended: 推荐列表（论文ID列表），按推荐顺序排列
            ground_truth: 真实标签集合（论文ID集合）
            k: 计算前K个推荐
            
        Returns:
            Recall@K值，范围在[0, 1]之间
            如果ground_truth为空，返回0.0
            如果k <= 0或recommended为空，返回0.0
        """
        if k <= 0:
            return 0.0
        
        if not recommended:
            return 0.0
        
        if not ground_truth:
            return 0.0
        
        # 取前k个推荐
        top_k = recommended[:k]
        
        # 计算交集：前k个推荐中属于真实标签的数量
        intersection = len(set(top_k) & ground_truth)
        
        # R@K = 交集大小 / 真实标签总数
        recall = intersection / len(ground_truth)
        
        return float(recall)
    
    @staticmethod
    def _average_precision(ranked: List[str], ground_truth: Set[str]) -> float:
        """计算单个查询的平均精度（Average Precision, AP）
        
        功能：
        - 计算单个查询的AP值
        - 公式: AP = (1/|R|) Σ_k P@k · rel(k)
        其中：
        - R: 真实标签集合
        - k: 排名位置
        - P@k: 前k个位置的精度
        - rel(k): 第k个位置是否相关（1或0）
        
        Args:
            ranked: 排序后的推荐列表（论文ID列表）
            ground_truth: 真实标签集合（论文ID集合）
            
        Returns:
            平均精度值，范围在[0, 1]之间
            如果ground_truth为空，返回0.0
        """
        if not ground_truth:
            return 0.0
        
        if not ranked:
            return 0.0
        
        hits = 0
        sum_precisions = 0.0
        
        for i, paper_id in enumerate(ranked, 1):
            if paper_id in ground_truth:
                hits += 1
                # 计算当前位置的精度
                precision_at_i = hits / i
                sum_precisions += precision_at_i
        
        # AP = 所有相关位置的精度之和 / 真实标签总数
        if hits == 0:
            return 0.0
        
        ap = sum_precisions / len(ground_truth)
        
        return float(ap)
    
    @staticmethod
    def mean_average_precision(
        ranked_results: List[List[str]],
        ground_truths: List[Set[str]]
    ) -> float:
        """计算平均精度均值（Mean Average Precision, MAP）
        
        功能：
        - 计算多个查询的平均精度均值
        - 公式: MAP = (1/|Q|) Σ_q AP_q
        其中：
        - Q: 查询集合
        - AP_q: 查询q的平均精度
        
        Args:
            ranked_results: 多个查询的排序结果列表，每个元素是一个排序后的推荐列表
            ground_truths: 多个查询的真实标签列表，每个元素是一个真实标签集合
            
        Returns:
            MAP值，范围在[0, 1]之间
            如果输入列表为空或长度不匹配，返回0.0
        """
        if not ranked_results or not ground_truths:
            return 0.0
        
        if len(ranked_results) != len(ground_truths):
            return 0.0
        
        aps = []
        for ranked, gt in zip(ranked_results, ground_truths):
            # 只对有ground_truth的查询计算AP（避免无ground_truth的查询拉低平均值）
            if not gt:
                continue
            ap = AccuracyMetrics._average_precision(ranked, gt)
            aps.append(ap)
        
        if not aps:
            return 0.0
        
        map_value = np.mean(aps)
        
        return float(map_value)
    
    @staticmethod
    def mean_reciprocal_rank(
        ranked_results: List[List[str]],
        ground_truths: List[Set[str]]
    ) -> float:
        """计算平均倒数排名（Mean Reciprocal Rank, MRR）
        
        功能：
        - 计算多个查询的平均倒数排名
        - 公式: MRR = (1/|Q|) Σ_q (1/rank_q)
        其中：
        - Q: 查询集合
        - rank_q: 查询q中第一个相关结果的排名位置
        
        Args:
            ranked_results: 多个查询的排序结果列表，每个元素是一个排序后的推荐列表
            ground_truths: 多个查询的真实标签列表，每个元素是一个真实标签集合
            
        Returns:
            MRR值，范围在[0, 1]之间
            如果输入列表为空或长度不匹配，返回0.0
        """
        if not ranked_results or not ground_truths:
            return 0.0
        
        if len(ranked_results) != len(ground_truths):
            return 0.0
        
        rrs = []
        for ranked, gt in zip(ranked_results, ground_truths):
            # 只对有ground_truth的查询计算RR（避免无ground_truth的查询拉低平均值）
            if not gt:
                continue
            # 找到第一个相关结果的排名
            reciprocal_rank = 0.0
            for i, paper_id in enumerate(ranked, 1):
                if paper_id in gt:
                    reciprocal_rank = 1.0 / i
                    break
            # 如果没有找到相关结果，reciprocal_rank保持为0.0
            rrs.append(reciprocal_rank)
        
        if not rrs:
            return 0.0
        
        mrr_value = np.mean(rrs)
        
        return float(mrr_value)

