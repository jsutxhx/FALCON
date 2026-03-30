"""多查询词分批次获取OpenCorpus数据"""
import json
import time
import sys
from pathlib import Path
from typing import List, Set, Dict, Any
from loguru import logger

# 添加项目根目录到路径，以便导入scripts模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data_acquisition import SemanticScholarAPI, fetch_opencorpus_data


def fetch_papers_with_multiple_queries(
    target_count: int = 7000,
    output_path: str = "data/raw/papers/opencorpus_papers.json",
    api_key: str = None,
    queries: List[str] = None,
    papers_per_query: int = 800
) -> None:
    """使用多个查询词分批次获取论文数据
    
    Args:
        target_count: 目标论文总数
        output_path: 输出文件路径
        api_key: API密钥
        queries: 查询词列表（如果为None，使用默认查询词列表）
        papers_per_query: 每个查询词获取的论文数量（默认800，受API限制）
    """
    logger.info(f"开始分批次获取论文，目标数量: {target_count}")
    
    # 默认查询词列表（计算机科学相关，选择更分散的关键词以减少重复）
    if queries is None:
        queries = [
            # 机器学习与AI（更具体的子领域）
            "transfer learning",
            "meta learning",
            "few shot learning",
            "adversarial machine learning",
            "federated learning",
            "graph neural networks",
            "attention mechanism",
            "transformer models",
            
            # 系统与网络（不同方向）
            "edge computing",
            "fog computing",
            "serverless computing",
            "container orchestration",
            "microservices architecture",
            "network function virtualization",
            "software defined networking",
            "network security protocols",
            
            # 数据与存储
            "time series analysis",
            "stream processing",
            "graph databases",
            "key value stores",
            "distributed storage",
            "data warehousing",
            "ETL pipelines",
            "data quality",
            
            # 理论与算法
            "approximation algorithms",
            "randomized algorithms",
            "online algorithms",
            "game theory",
            "combinatorial optimization",
            "linear programming",
            "graph algorithms",
            "string algorithms",
            
            # 应用领域
            "recommender systems",
            "search engines",
            "social networks analysis",
            "computational finance",
            "computational chemistry",
            "computational physics",
            "digital signal processing",
            "image processing",
            
            # 软件工程（更具体）
            "code refactoring",
            "static analysis tools",
            "continuous integration",
            "agile methodologies",
            "design patterns",
            "software metrics",
            "code review",
            "technical debt",
            
            # 新兴技术
            "edge AI",
            "neuromorphic computing",
            "homomorphic encryption",
            "differential privacy",
            "explainable AI",
            "automated machine learning",
            "neural architecture search",
            "model compression"
        ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 存储所有论文（使用id作为key去重）
    all_papers_dict: Dict[str, Dict[str, Any]] = {}
    
    # 如果输出文件已存在，先加载已有数据
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_papers = json.load(f)
            for paper in existing_papers:
                paper_id = paper.get("id")
                if paper_id:
                    all_papers_dict[paper_id] = paper
            logger.info(f"检测到已有数据文件，加载了 {len(all_papers_dict)} 篇论文")
        except Exception as e:
            logger.warning(f"加载已有数据失败: {e}，将从头开始")
    
    # 计算每个查询词需要获取的数量
    papers_per_query = min(papers_per_query, 800)  # API限制：最多800篇
    
    # 考虑到去重，需要更多的查询词
    # 假设去重率约30-40%，所以需要更多查询词
    estimated_dedup_rate = 0.35  # 假设35%的重复率
    effective_papers_per_query = int(papers_per_query * (1 - estimated_dedup_rate))
    
    # 计算还需要多少论文
    remaining_needed = max(0, target_count - len(all_papers_dict))
    if remaining_needed > 0:
        num_queries_needed = (remaining_needed + effective_papers_per_query - 1) // effective_papers_per_query
        # 至少使用足够的查询词，最多使用所有可用的查询词
        num_queries_needed = min(num_queries_needed, len(queries))
        queries_to_use = queries[:num_queries_needed]
        logger.info(f"当前已有 {len(all_papers_dict)} 篇，还需要 {remaining_needed} 篇")
        logger.info(f"将使用 {len(queries_to_use)} 个查询词，每个查询词最多获取 {papers_per_query} 篇论文")
        logger.info(f"考虑到去重（估计重复率{estimated_dedup_rate*100:.0f}%），每个查询词有效约 {effective_papers_per_query} 篇")
    else:
        logger.info(f"已达到目标数量 {target_count} 篇，无需继续获取")
        queries_to_use = []
    
    # 对每个查询词进行数据获取
    for i, query in enumerate(queries_to_use, 1):
        if len(all_papers_dict) >= target_count:
            logger.info(f"已达到目标数量 {target_count} 篇，停止获取")
            break
        
        logger.info(f"=" * 60)
        logger.info(f"批次 {i}/{len(queries_to_use)}: 查询词 '{query}'")
        logger.info(f"当前已有论文: {len(all_papers_dict)} 篇")
        logger.info(f"=" * 60)
        
        # 计算这个查询词需要获取的数量
        remaining = target_count - len(all_papers_dict)
        batch_target = min(papers_per_query, remaining)
        
        try:
            # 创建临时文件路径
            temp_file = output_path.parent / f"temp_{i}_{query.replace(' ', '_')}.json"
            
            # 使用单个查询词获取数据
            fetch_opencorpus_data(
                target_count=batch_target,
                output_path=str(temp_file),
                api_key=api_key,
                query=query
            )
            
            # 读取获取的数据
            if temp_file.exists():
                with open(temp_file, 'r', encoding='utf-8') as f:
                    batch_papers = json.load(f)
                
                # 合并到总数据中（去重）
                new_count = 0
                for paper in batch_papers:
                    paper_id = paper.get("id")
                    if paper_id and paper_id not in all_papers_dict:
                        all_papers_dict[paper_id] = paper
                        new_count += 1
                
                logger.info(f"批次 {i} 完成: 获取 {len(batch_papers)} 篇，新增 {new_count} 篇（去重后），总计 {len(all_papers_dict)} 篇")
                
                # 删除临时文件
                temp_file.unlink()
                
                # 每批次保存一次（防止数据丢失）
                all_papers = list(all_papers_dict.values())
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_papers, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存中间结果: {len(all_papers)} 篇论文")
            else:
                logger.warning(f"批次 {i} 未生成数据文件，跳过")
        
        except Exception as e:
            logger.error(f"批次 {i} 失败: {e}")
            logger.warning("继续下一个批次...")
            continue
        
        # 批次之间等待一段时间（避免速率限制）
        if i < len(queries_to_use):
            logger.info(f"等待5秒后继续下一个批次...")
            time.sleep(5)
    
    # 保存最终结果
    all_papers = list(all_papers_dict.values())
    logger.info(f"=" * 60)
    logger.info(f"所有批次完成！")
    logger.info(f"总计获取: {len(all_papers)} 篇论文（去重后）")
    logger.info(f"=" * 60)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已保存到: {output_path}")
    
    if len(all_papers) < target_count:
        logger.warning(f"未达到目标数量 {target_count} 篇，实际获取 {len(all_papers)} 篇")
        logger.warning("可能的原因：")
        logger.warning("1. API速率限制导致部分批次失败")
        logger.warning("2. 查询词结果不足")
        logger.warning("3. 去重后数量减少")
        logger.warning("建议：可以添加更多查询词或使用其他数据源")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用多个查询词分批次获取OpenCorpus数据")
    parser.add_argument(
        "--target_count",
        type=int,
        default=7000,
        help="目标论文总数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/papers/opencorpus_papers.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Semantic Scholar API密钥"
    )
    parser.add_argument(
        "--papers_per_query",
        type=int,
        default=800,
        help="每个查询词获取的论文数量（默认800，受API限制）"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="查询词列表（可选，如果不提供则使用默认列表）"
    )
    
    args = parser.parse_args()
    
    fetch_papers_with_multiple_queries(
        target_count=args.target_count,
        output_path=args.output,
        api_key=args.api_key,
        queries=args.queries,
        papers_per_query=args.papers_per_query
    )

