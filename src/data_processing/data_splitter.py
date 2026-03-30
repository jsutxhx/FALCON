"""数据集划分模块"""
import random
from typing import Dict, List, Tuple
from src.data_processing.data_structures import Paper, Citation


def split_data(
    papers: List[Paper],
    citations: List[Citation],
    ratios: Dict[str, float],
    random_seed: int = 42
) -> Dict[str, Dict[str, List]]:
    """将论文和引用划分为训练集、验证集和测试集
    
    功能：
    - 按比例随机划分论文
    - 根据论文划分分配引用（确保引用的source和target在同一集合中）
    - 避免数据泄露
    
    Args:
        papers: 论文列表
        citations: 引用列表
        ratios: 划分比例，格式如 {"train": 0.8, "val": 0.1, "test": 0.1}
        random_seed: 随机种子
        
    Returns:
        包含三个集合的字典，格式为:
        {
            "train": {"papers": [...], "citations": [...]},
            "val": {"papers": [...], "citations": [...]},
            "test": {"papers": [...], "citations": [...]}
        }
        
    Raises:
        ValueError: 如果比例之和不为1.0，或比例值无效
    """
    # 验证比例
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1.0，当前为{total_ratio}")
    
    for split_name, ratio in ratios.items():
        if ratio < 0 or ratio > 1:
            raise ValueError(f"比例值必须在[0, 1]范围内，{split_name}={ratio}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建论文ID到论文的映射
    paper_dict = {paper.id: paper for paper in papers}
    
    # 随机打乱论文列表
    shuffled_papers = papers.copy()
    random.shuffle(shuffled_papers)
    
    # 计算每个集合的论文数量
    n_papers = len(papers)
    n_train = int(n_papers * ratios["train"])
    n_val = int(n_papers * ratios["val"])
    # n_test = n_papers - n_train - n_val  # 确保所有论文都被分配
    
    # 划分论文
    train_papers = shuffled_papers[:n_train]
    val_papers = shuffled_papers[n_train:n_train + n_val]
    test_papers = shuffled_papers[n_train + n_val:]
    
    # 创建论文ID到集合的映射
    paper_to_split = {}
    for paper in train_papers:
        paper_to_split[paper.id] = "train"
    for paper in val_papers:
        paper_to_split[paper.id] = "val"
    for paper in test_papers:
        paper_to_split[paper.id] = "test"
    
    # 初始化各集合的引用列表
    train_citations = []
    val_citations = []
    test_citations = []
    
    # 分配引用：只有当source和target都在同一集合时，才将该引用分配到该集合
    # 这确保了严格的数据划分，避免数据泄露
    for citation in citations:
        source_split = paper_to_split.get(citation.source_paper_id)
        target_split = paper_to_split.get(citation.target_paper_id)
        
        # 如果source或target不在任何集合中，跳过该引用
        if source_split is None or target_split is None:
            continue
        
        # 只有当source和target在同一集合时，才分配该引用
        # 这确保了严格的数据划分，避免跨集合的数据泄露
        if source_split == target_split:
            if source_split == "train":
                train_citations.append(citation)
            elif source_split == "val":
                val_citations.append(citation)
            elif source_split == "test":
                test_citations.append(citation)
        # 如果source和target不在同一集合，跳过该引用以避免数据泄露
    
    # 构建返回结果
    result = {
        "train": {
            "papers": train_papers,
            "citations": train_citations
        },
        "val": {
            "papers": val_papers,
            "citations": val_citations
        },
        "test": {
            "papers": test_papers,
            "citations": test_citations
        }
    }
    
    return result

