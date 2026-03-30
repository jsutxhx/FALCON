"""数据预处理脚本"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from dataclasses import asdict
from loguru import logger

from src.data_processing.data_structures import Paper, Citation
from src.data_processing.text_preprocessor import TextPreprocessor


def clean_text(text: str) -> str:
    """清洗文本"""
    if not text:
        return ""
    
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 统一Unicode字符
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text.strip()


def preprocess_papers(
    input_path: str,
    output_path: str,
    min_abstract_length: int = 50
) -> List[Paper]:
    """预处理论文数据
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        min_abstract_length: 最小摘要长度
        
    Returns:
        清洗后的论文列表
    """
    logger.info(f"开始预处理论文数据: {input_path}")
    
    # 加载原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_papers = json.load(f)
    
    logger.info(f"加载了 {len(raw_papers)} 篇原始论文")
    
    # 去重（基于ID）
    seen_ids = set()
    unique_papers = []
    for paper_data in raw_papers:
        paper_id = paper_data.get("id") or paper_data.get("paperId")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            unique_papers.append(paper_data)
    
    logger.info(f"去重后剩余 {len(unique_papers)} 篇论文")
    
    # 过滤和清洗
    cleaned_papers = []
    preprocessor = TextPreprocessor()
    
    for paper_data in unique_papers:
        # 检查必需字段
        paper_id = paper_data.get("id") or paper_data.get("paperId")
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")
        
        if not paper_id or not title:
            continue
        
        # 过滤无摘要或摘要太短的论文
        if not abstract or len(abstract) < min_abstract_length:
            continue
        
        # 清洗文本
        title = clean_text(title)
        abstract = clean_text(abstract)
        
        # 处理作者
        authors = paper_data.get("authors", [])
        if isinstance(authors, list):
            authors = [clean_text(str(author)) for author in authors if author]
        else:
            authors = []
        
        # 处理年份
        year = paper_data.get("year")
        if year:
            try:
                year = int(year)
            except:
                year = None
        
        # 创建Paper对象
        paper = Paper(
            id=str(paper_id),
            title=title,
            abstract=abstract,
            authors=authors,
            year=year or 2000,
            venue=clean_text(paper_data.get("venue", "")),
            citation_count=paper_data.get("citationCount", paper_data.get("citation_count", 0)),
            doi=paper_data.get("doi")
        )
        
        cleaned_papers.append(paper)
    
    logger.info(f"清洗后剩余 {len(cleaned_papers)} 篇论文")
    
    # 保存清洗后的论文
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    papers_dict = [asdict(paper) for paper in cleaned_papers]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"清洗后的论文已保存到: {output_path}")
    
    return cleaned_papers


def build_citation_graph(
    papers: List[Paper],
    output_path: str
) -> List[Citation]:
    """构建引用图
    
    Args:
        papers: 论文列表
        output_path: 输出文件路径
        
    Returns:
        引用列表
    """
    logger.info("开始构建引用图")
    
    # 创建论文ID到Paper对象的映射
    paper_dict = {paper.id: paper for paper in papers}
    
    citations = []
    citation_count = 0
    
    for paper in papers:
        # 从原始数据中获取引用列表（如果存在）
        if hasattr(paper, 'references') and paper.references:
            for ref_data in paper.references:
                if isinstance(ref_data, dict):
                    target_id = ref_data.get("paper_id") or ref_data.get("paperId")
                else:
                    target_id = str(ref_data)
                
                # 验证引用有效性
                if target_id and target_id in paper_dict:
                    citation = Citation(
                        source_paper_id=paper.id,
                        target_paper_id=target_id,
                        context="",  # 原始数据中可能没有上下文
                        position="introduction"  # 默认位置
                    )
                    citations.append(citation)
                    citation_count += 1
    
    logger.info(f"构建了 {len(citations)} 条有效引用")
    
    # 保存引用数据
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    citations_dict = [asdict(citation) for citation in citations]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(citations_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"引用数据已保存到: {output_path}")
    
    return citations


def split_data(
    papers: List[Paper],
    citations: List[Citation],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> None:
    """划分数据集
    
    Args:
        papers: 论文列表
        citations: 引用列表
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    logger.info("开始划分数据集")
    
    import random
    random.seed(42)
    
    # 打乱论文顺序
    shuffled_papers = papers.copy()
    random.shuffle(shuffled_papers)
    
    # 计算划分点
    n = len(shuffled_papers)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_papers = shuffled_papers[:train_end]
    val_papers = shuffled_papers[train_end:val_end]
    test_papers = shuffled_papers[val_end:]
    
    logger.info(f"训练集: {len(train_papers)} 篇")
    logger.info(f"验证集: {len(val_papers)} 篇")
    logger.info(f"测试集: {len(test_papers)} 篇")
    
    # 创建论文ID集合
    train_ids = {p.id for p in train_papers}
    val_ids = {p.id for p in val_papers}
    test_ids = {p.id for p in test_papers}
    
    # 划分引用（确保引用的两端在同一集合中）
    train_citations = []
    val_citations = []
    test_citations = []
    
    for citation in citations:
        source_id = citation.source_paper_id
        target_id = citation.target_paper_id
        
        if source_id in train_ids and target_id in train_ids:
            train_citations.append(citation)
        elif source_id in val_ids and target_id in val_ids:
            val_citations.append(citation)
        elif source_id in test_ids and target_id in test_ids:
            test_citations.append(citation)
    
    logger.info(f"训练集引用: {len(train_citations)} 条")
    logger.info(f"验证集引用: {len(val_citations)} 条")
    logger.info(f"测试集引用: {len(test_citations)} 条")
    
    # 保存划分结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("train", train_papers, train_citations),
        ("val", val_papers, val_citations),
        ("test", test_papers, test_citations)
    ]
    
    for split_name, split_papers, split_citations in splits:
        # 保存论文
        papers_file = output_dir / f"{split_name}_papers.json"
        papers_dict = [asdict(p) for p in split_papers]
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(papers_dict, f, ensure_ascii=False, indent=2)
        
        # 保存引用
        citations_file = output_dir / f"{split_name}_citations.json"
        citations_dict = [asdict(c) for c in split_citations]
        with open(citations_file, 'w', encoding='utf-8') as f:
            json.dump(citations_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{split_name}集已保存到: {papers_file} 和 {citations_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="输出目录"
    )
    parser.add_argument(
        "--min_abstract_length",
        type=int,
        default=50,
        help="最小摘要长度"
    )
    
    args = parser.parse_args()
    
    # Step 1: 预处理论文
    cleaned_papers = preprocess_papers(
        input_path=args.input,
        output_path=f"{args.output_dir}/cleaned_papers.json",
        min_abstract_length=args.min_abstract_length
    )
    
    # Step 2: 构建引用图
    citations = build_citation_graph(
        papers=cleaned_papers,
        output_path=f"{args.output_dir}/citations.json"
    )
    
    # Step 3: 划分数据集
    split_data(
        papers=cleaned_papers,
        citations=citations,
        output_dir=f"{args.output_dir}/../splits"
    )
    
    logger.info("数据预处理完成")


if __name__ == "__main__":
    main()

