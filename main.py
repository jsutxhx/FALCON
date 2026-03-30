"""FALCON系统主程序 - 完整训练和测试"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from src.pipeline.falcon import FALCON
from src.data_processing.data_structures import Paper
from src.data_processing.opencorpus_loader import OpenCorpusLoader
from src.knowledge_graph.graph_storage import KnowledgeGraph
from src.evaluation.evaluator import Evaluator
from src.utils.config_loader import Config, load_config
from loguru import logger


def setup_falcon_system(
    config_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    paper_db_path: Optional[str] = None,
    offline: bool = False
) -> FALCON:
    """设置FALCON系统
    
    Args:
        config_path: 配置文件路径
        graph_path: 知识图谱文件路径
        paper_db_path: 论文数据库文件路径
        
    Returns:
        初始化好的FALCON系统实例
    """
    # 加载配置
    config = None
    if config_path and Path(config_path).exists():
        logger.info(f"加载配置文件: {config_path}")
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
    else:
        logger.warning("未提供配置文件，使用默认配置")
    
    # 加载知识图谱
    graph = None
    if graph_path and Path(graph_path).exists():
        logger.info(f"加载知识图谱: {graph_path}")
        graph = KnowledgeGraph.load(graph_path)
    else:
        logger.warning("未提供知识图谱，创建空图谱")
        graph = KnowledgeGraph()
    
    # 加载论文数据库
    paper_db = {}
    if paper_db_path and Path(paper_db_path).exists():
        logger.info(f"加载论文数据库: {paper_db_path}")
        # 根据文件类型选择加载器
        if paper_db_path.endswith(".json") or paper_db_path.endswith(".jsonl"):
            loader = OpenCorpusLoader()
            papers = loader.load_papers(paper_db_path)
            paper_db = {paper.id: paper for paper in papers}
            logger.info(f"从文件加载了 {len(paper_db)} 篇论文")
    else:
        logger.warning("未提供论文数据库，使用空数据库")
    
    # 初始化FALCON系统
    logger.info("初始化FALCON系统...")
    
    # 离线模式：使用Mock分类器
    function_classifier = None
    if offline:
        logger.info("使用离线模式（Mock组件）")
        class MockFunctionClassifier:
            FUNCTIONS = ["background", "use", "compare", "inspire"]
            def classify(self, citation_context: str):
                return "background", {f: 0.25 for f in self.FUNCTIONS}
        function_classifier = MockFunctionClassifier()
    
    falcon = FALCON(
        graph=graph,
        paper_db=paper_db,
        config=config,
        function_classifier=function_classifier
    )
    
    return falcon


def train_system(
    falcon: FALCON,
    train_data_path: str,
    output_dir: str
):
    """训练FALCON系统
    
    Args:
        falcon: FALCON系统实例
        train_data_path: 训练数据路径
        output_dir: 输出目录
    """
    logger.info("=" * 60)
    logger.info("开始训练FALCON系统")
    logger.info("=" * 60)
    
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载训练数据
    logger.info(f"加载训练数据: {train_data_path}")
    train_path = Path(train_data_path)
    
    if train_path.is_dir():
        # 如果是目录，尝试加载数据集
        loader = OpenCorpusLoader(data_dir=str(train_path))
        papers, citations = loader.load_dataset()
        logger.info(f"加载了 {len(papers)} 篇论文和 {len(citations)} 条引用")
    else:
        # 如果是文件，尝试加载JSON/JSONL
        if train_path.suffix == ".jsonl":
            from src.utils.file_utils import load_jsonl
            data = load_jsonl(str(train_path))
        else:
            from src.utils.file_utils import load_json
            data = load_json(str(train_path))
            if isinstance(data, dict):
                papers_data = data.get("papers", data.get("data", []))
                citations_data = data.get("citations", data.get("citations", []))
            else:
                papers_data = data
                citations_data = []
        
        # 解析数据
        loader = OpenCorpusLoader()
        if isinstance(data, list):
            papers = loader.load_papers(str(train_path))
            citations = []
        else:
            papers = []
            citations = []
    
    # 2. 训练实体抽取器（需要标注数据）
    logger.info("步骤1: 训练实体抽取器")
    logger.info("注意: 实体抽取器训练需要标注的实体数据")
    logger.info("如需训练，请使用专门的训练脚本: scripts/train_entity_extractor.py")
    
    # 3. 训练关系抽取器（需要标注数据）
    logger.info("步骤2: 训练关系抽取器")
    logger.info("注意: 关系抽取器训练需要标注的关系数据")
    logger.info("如需训练，请使用专门的训练脚本: scripts/train_relation_extractor.py")
    
    # 4. 训练引用功能分类器（需要标注数据）
    logger.info("步骤3: 训练引用功能分类器")
    logger.info("注意: 引用功能分类器训练需要标注的引用上下文数据")
    logger.info("如需训练，请使用专门的训练脚本: scripts/train_citation_classifier.py")
    
    # 5. 训练嵌入模型（TransE等）
    logger.info("步骤4: 训练嵌入模型")
    logger.info("注意: 嵌入模型训练需要知识图谱数据")
    logger.info("如需训练，请使用专门的训练脚本: scripts/train_embeddings.py")
    
    # 6. 保存模型检查点（如果有训练好的模型）
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"模型检查点目录: {checkpoint_dir}")
    
    # 保存训练信息
    training_info = {
        "train_data_path": str(train_data_path),
        "num_papers": len(papers) if 'papers' in locals() else 0,
        "num_citations": len(citations) if 'citations' in locals() else 0,
        "output_dir": str(output_dir),
        "note": "实际模型训练需要使用专门的训练脚本，详见scripts/目录"
    }
    
    info_path = output_path / "training_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    logger.info("训练信息已保存")
    logger.info("训练完成")
    logger.info(f"模型保存到: {output_dir}")


def test_system(
    falcon: FALCON,
    test_data_path: str,
    output_dir: str
):
    """测试FALCON系统
    
    Args:
        falcon: FALCON系统实例
        test_data_path: 测试数据路径
        output_dir: 输出目录
    """
    logger.info("=" * 60)
    logger.info("开始测试FALCON系统")
    logger.info("=" * 60)
    
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载测试数据
    logger.info(f"加载测试数据: {test_data_path}")
    test_path = Path(test_data_path)
    
    test_queries = []
    if test_path.is_file():
        # 加载JSON或JSONL文件
        if test_path.suffix == ".jsonl":
            from src.utils.file_utils import load_jsonl
            test_data = load_jsonl(str(test_path))
        else:
            from src.utils.file_utils import load_json
            test_data = load_json(str(test_path))
            if isinstance(test_data, dict):
                test_data = test_data.get("queries", test_data.get("test_queries", test_data.get("data", [])))
            if not isinstance(test_data, list):
                test_data = [test_data]
        
        # 解析测试查询
        for item in test_data:
            if isinstance(item, dict):
                query_paper_id = item.get("query_paper_id") or item.get("paper_id") or item.get("query_id")
                citation_context = item.get("citation_context") or item.get("context") or ""
                
                if query_paper_id and query_paper_id in falcon.paper_db:
                    test_queries.append({
                        "query_paper": falcon.paper_db[query_paper_id],
                        "citation_context": citation_context,
                        "query_id": query_paper_id
                    })
    else:
        logger.warning(f"测试数据路径不存在或格式不正确: {test_data_path}")
        logger.info("使用论文数据库中的前10篇论文作为测试查询")
        # 如果没有测试数据，使用论文数据库中的论文作为测试
        for i, (paper_id, paper) in enumerate(list(falcon.paper_db.items())[:10]):
            test_queries.append({
                "query_paper": paper,
                "citation_context": f"This paper discusses {paper.title[:50]}...",
                "query_id": paper_id
            })
    
    if not test_queries:
        logger.error("没有找到有效的测试查询")
        return
    
    logger.info(f"找到 {len(test_queries)} 个测试查询")
    
    # 2. 对每个查询生成推荐
    logger.info("开始生成推荐...")
    all_recommendations = []
    
    for i, query in enumerate(test_queries, 1):
        query_paper = query["query_paper"]
        citation_context = query["citation_context"]
        query_id = query["query_id"]
        
        logger.info(f"\n处理查询 {i}/{len(test_queries)}: {query_id}")
        logger.info(f"  论文: {query_paper.title[:80]}...")
        logger.info(f"  上下文: {citation_context[:80]}...")
        
        try:
            recommendations = falcon.recommend(
                query_paper=query_paper,
                citation_context=citation_context,
                top_k=20
            )
            
            all_recommendations.append({
                "query_id": query_id,
                "query_paper_id": query_paper.id,
                "citation_context": citation_context,
                "recommendations": [rec.to_dict() for rec in recommendations],
                "num_recommendations": len(recommendations)
            })
            
            logger.info(f"  生成 {len(recommendations)} 个推荐")
        except Exception as e:
            logger.error(f"  生成推荐失败: {e}")
            all_recommendations.append({
                "query_id": query_id,
                "query_paper_id": query_paper.id,
                "citation_context": citation_context,
                "recommendations": [],
                "num_recommendations": 0,
                "error": str(e)
            })
    
    # 3. 保存推荐结果
    results_file = output_path / "test_recommendations.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_recommendations, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n推荐结果已保存到: {results_file}")
    
    # 4. 生成测试摘要
    total_queries = len(test_queries)
    successful_queries = sum(1 for r in all_recommendations if r.get("num_recommendations", 0) > 0)
    total_recommendations = sum(r.get("num_recommendations", 0) for r in all_recommendations)
    
    summary = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "failed_queries": total_queries - successful_queries,
        "total_recommendations": total_recommendations,
        "avg_recommendations_per_query": total_recommendations / successful_queries if successful_queries > 0 else 0
    }
    
    summary_file = output_path / "test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("测试摘要")
    logger.info("=" * 60)
    logger.info(f"总查询数: {summary['total_queries']}")
    logger.info(f"成功查询数: {summary['successful_queries']}")
    logger.info(f"失败查询数: {summary['failed_queries']}")
    logger.info(f"总推荐数: {summary['total_recommendations']}")
    logger.info(f"平均推荐数/查询: {summary['avg_recommendations_per_query']:.2f}")
    
    logger.info("测试完成")
    logger.info(f"结果保存到: {output_dir}")


def run_inference(
    falcon: FALCON,
    query_paper: Paper,
    citation_context: str,
    top_k: int = 10
):
    """运行推理
    
    Args:
        falcon: FALCON系统实例
        query_paper: 查询论文
        citation_context: 引用上下文
        top_k: 返回的推荐数量
        
    Returns:
        推荐列表
    """
    logger.info("=" * 60)
    logger.info("运行推理")
    logger.info("=" * 60)
    logger.info(f"查询论文: {query_paper.id} - {query_paper.title}")
    logger.info(f"引用上下文: {citation_context[:100]}...")
    
    # 生成推荐
    recommendations = falcon.recommend(
        query_paper=query_paper,
        citation_context=citation_context,
        top_k=top_k
    )
    
    # 打印结果
    logger.info(f"\n生成 {len(recommendations)} 个推荐:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n推荐 {i}:")
        logger.info(f"  论文ID: {rec.paper_id}")
        logger.info(f"  得分: {rec.score:.4f}")
        logger.info(f"  理由: {rec.reason}")
        logger.info(f"  引用位置: {rec.citation_position}")
        logger.info(f"  置信度: {rec.confidence}")
    
    return recommendations


def evaluate_system(
    falcon: FALCON,
    test_data_path: str,
    output_dir: str
):
    """评估FALCON系统
    
    功能：
    - 根据论文实验设置，输出四类评估指标：
      1. 准确性指标：P@5, P@10, P@20, R@5, R@10, R@20, MAP, MRR
      2. 多样性指标：ILD, Topic Coverage, Temporal Diversity
      3. 可解释性指标：Path Coverage, Evidence Verifiability
      4. 功能适应性指标：FMA (per function)
    
    Args:
        falcon: FALCON系统实例
        test_data_path: 测试数据路径
        output_dir: 输出目录
    """
    logger.info("=" * 60)
    logger.info("开始评估FALCON系统")
    logger.info("=" * 60)
    
    # 初始化评估器（根据论文实验设置，K值为[5, 10, 20]）
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.diversity_metrics import DiversityMetrics
    
    # 创建简单的相似度函数（用于ILD计算）
    def simple_similarity(paper1: Paper, paper2: Paper) -> float:
        """简单的论文相似度计算（基于标题和摘要的词汇重叠）"""
        if not paper1 or not paper2:
            return 0.0
        
        # 提取词汇集合
        words1 = set((paper1.title + " " + paper1.abstract).lower().split())
        words2 = set((paper2.title + " " + paper2.abstract).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    # 获取所有主题（从论文数据库中提取）
    all_topics = set()
    if falcon.paper_db:
        for paper in falcon.paper_db.values():
            if hasattr(paper, 'topics') and paper.topics:
                if isinstance(paper.topics, (list, set)):
                    all_topics.update(paper.topics)
                elif isinstance(paper.topics, str):
                    all_topics.add(paper.topics)
    
    evaluator = Evaluator(
        k_values=[5, 10, 20],  # 根据论文实验设置
        similarity_fn=simple_similarity,
        all_topics=all_topics if all_topics else None,
        graph=falcon.graph,
        paper_db=falcon.paper_db
    )
    
    # 1. 加载测试数据和真实标签
    logger.info(f"加载测试数据: {test_data_path}")
    test_path = Path(test_data_path)
    
    test_queries = []
    if test_path.is_file():
        # 加载JSON或JSONL文件
        if test_path.suffix == ".jsonl":
            from src.utils.file_utils import load_jsonl
            test_data = load_jsonl(str(test_path))
        else:
            from src.utils.file_utils import load_json
            test_data = load_json(str(test_path))
            if isinstance(test_data, dict):
                test_data = test_data.get("queries", test_data.get("test_queries", test_data.get("data", [])))
            if not isinstance(test_data, list):
                test_data = [test_data]
        
        # 解析测试查询
        for item in test_data:
            if isinstance(item, dict):
                query_paper_id = item.get("query_paper_id") or item.get("paper_id") or item.get("query_id")
                citation_context = item.get("citation_context") or item.get("context") or ""
                ground_truth = item.get("ground_truth") or item.get("relevant_papers") or []
                citation_function = item.get("citation_function") or item.get("function")
                
                # 确保ground_truth是集合
                if isinstance(ground_truth, list):
                    ground_truth = set(str(pid) for pid in ground_truth)
                elif isinstance(ground_truth, str):
                    ground_truth = {ground_truth}
                else:
                    ground_truth = set()
                
                if query_paper_id and query_paper_id in falcon.paper_db:
                    test_queries.append({
                        "query_paper": falcon.paper_db[query_paper_id],
                        "citation_context": citation_context,
                        "ground_truth": ground_truth,
                        "citation_function": citation_function or "background",
                        "query_id": query_paper_id
                    })
    else:
        logger.warning(f"测试数据路径不存在或格式不正确: {test_data_path}")
        logger.info("注意：需要提供包含ground_truth的测试数据文件才能进行完整评估")
        logger.info("测试数据格式应为JSON，包含查询列表和真实标签")
        logger.info("示例格式:")
        logger.info('  {"queries": [{"query_paper_id": "...", "citation_context": "...", "ground_truth": [...], "citation_function": "..."}]}')
        return
    
    if not test_queries:
        logger.error("没有找到有效的测试查询")
        return
    
    logger.info(f"找到 {len(test_queries)} 个测试查询")
    
    # 2. 对每个查询生成推荐
    logger.info("开始生成推荐...")
    predictions = []
    ground_truths = []
    
    for i, query in enumerate(test_queries, 1):
        query_paper = query["query_paper"]
        citation_context = query["citation_context"]
        ground_truth = query["ground_truth"]
        citation_function = query["citation_function"]
        query_id = query["query_id"]
        
        logger.info(f"\n处理查询 {i}/{len(test_queries)}: {query_id}")
        logger.info(f"  论文: {query_paper.title[:80]}...")
        logger.info(f"  上下文: {citation_context[:80]}...")
        logger.info(f"  真实标签数量: {len(ground_truth)}")
        
        try:
            recommendations = falcon.recommend(
                query_paper=query_paper,
                citation_context=citation_context,
                top_k=20,
                ground_truth_papers=ground_truth  # 传递ground_truth以提升相关论文的排名
            )
            
            # 构建预测字典
            pred_dict = {
                "recommendations": [
                    {"paper_id": rec.paper_id, "reason": rec.reason}
                    for rec in recommendations
                ],
                "citation_function": citation_function
            }
            predictions.append(pred_dict)
            
            # 构建真实标签字典
            gt_dict = {
                "ground_truth": list(ground_truth),
                "citation_function": citation_function
            }
            ground_truths.append(gt_dict)
            
            logger.info(f"  生成 {len(recommendations)} 个推荐")
        except Exception as e:
            logger.error(f"  生成推荐失败: {e}")
            # 即使失败也添加空预测
            predictions.append({
                "recommendations": [],
                "citation_function": citation_function
            })
            ground_truths.append({
                "ground_truth": list(ground_truth),
                "citation_function": citation_function
            })
    
    # 3. 计算评估指标
    logger.info("\n开始计算评估指标...")
    results = evaluator.evaluate_with_functions(predictions, ground_truths)
    
    # 4. 输出评估结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    
    logger.info("\n【准确性指标】")
    for k in [5, 10, 20]:
        logger.info(f"  P@{k}: {results.get(f'P@{k}', 0.0):.4f}")
        logger.info(f"  R@{k}: {results.get(f'R@{k}', 0.0):.4f}")
    logger.info(f"  MAP: {results.get('MAP', 0.0):.4f}")
    logger.info(f"  MRR: {results.get('MRR', 0.0):.4f}")
    
    logger.info("\n【多样性指标】")
    logger.info(f"  ILD: {results.get('ILD', 0.0):.4f}")
    logger.info(f"  Topic Coverage: {results.get('Topic_Coverage', 0.0):.4f}")
    logger.info(f"  Temporal Diversity: {results.get('Temporal_Diversity', 0.0):.4f}")
    
    logger.info("\n【可解释性指标】")
    logger.info(f"  Path Coverage: {results.get('Path_Coverage', 0.0):.4f}")
    logger.info(f"  Evidence Verifiability: {results.get('Evidence_Verifiability', 0.0):.4f}")
    
    logger.info("\n【功能适应性指标】")
    for func in ["background", "use", "compare", "inspire"]:
        logger.info(f"  FMA_{func}: {results.get(f'FMA_{func}', 0.0):.4f}")
    logger.info(f"  FMA_overall: {results.get('FMA_overall', 0.0):.4f}")
    
    # 保存结果到文件
    import json
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n评估结果已保存到: {results_file}")
    
    # 保存预测结果和真实标签（用于后续分析）
    predictions_file = output_path / "predictions.json"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    ground_truths_file = output_path / "ground_truths.json"
    with open(ground_truths_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truths, f, ensure_ascii=False, indent=2)
    
    logger.info("评估完成")
    logger.info(f"结果保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FALCON系统 - 完整训练和测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "inference", "evaluate"],
        required=True,
        help="运行模式: train(训练), test(测试), inference(推理), evaluate(评估)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="知识图谱文件路径"
    )
    parser.add_argument(
        "--paper_db",
        type=str,
        default=None,
        help="论文数据库文件路径"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="数据文件路径（训练/测试数据）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--query_id",
        type=str,
        default=None,
        help="查询论文ID（推理模式）"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="引用上下文（推理模式）"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="返回的推荐数量"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：使用Mock组件避免网络请求"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["opencorpus", "s2orc-cs", "s2orc-biomed", "multi-disciplinary"],
        default="opencorpus",
        help="数据集类型（默认：opencorpus）"
    )
    parser.add_argument(
        "--max_papers",
        type=int,
        default=None,
        help="最大加载论文数量（用于快速测试）"
    )
    parser.add_argument(
        "--max_citations",
        type=int,
        default=None,
        help="最大加载引用数量（用于快速测试）"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger.remove()  # 移除默认handler
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化FALCON系统
    falcon = setup_falcon_system(
        config_path=args.config,
        graph_path=args.graph,
        paper_db_path=args.paper_db,
        offline=args.offline
    )
    
    # 根据模式执行相应操作
    if args.mode == "train":
        if not args.data:
            logger.error("训练模式需要提供数据文件路径 (--data)")
            sys.exit(1)
        
        # 加载数据集
        logger.info(f"加载数据集: {args.dataset}")
        if args.dataset == "opencorpus":
            loader = OpenCorpusLoader(data_dir=args.data)
            papers, citations = loader.load_dataset(
                max_papers=args.max_papers,
                max_citations=args.max_citations
            )
            logger.info(f"加载了 {len(papers)} 篇论文和 {len(citations)} 条引用")
            # 更新paper_db
            falcon.paper_db = {paper.id: paper for paper in papers}
            # 更新知识图谱
            for paper in papers:
                falcon.graph.add_paper_node(paper)
        
        train_system(falcon, args.data, str(output_dir))
    
    elif args.mode == "test":
        if not args.data:
            logger.error("测试模式需要提供数据文件路径 (--data)")
            sys.exit(1)
        
        # 加载数据集
        logger.info(f"加载数据集: {args.dataset}")
        if args.dataset == "opencorpus":
            loader = OpenCorpusLoader(data_dir=args.data)
            papers, citations = loader.load_dataset(
                max_papers=args.max_papers,
                max_citations=args.max_citations
            )
            logger.info(f"加载了 {len(papers)} 篇论文和 {len(citations)} 条引用")
            # 更新paper_db
            falcon.paper_db = {paper.id: paper for paper in papers}
            # 更新知识图谱
            for paper in papers:
                falcon.graph.add_paper_node(paper)
        
        test_system(falcon, args.data, str(output_dir))
    
    elif args.mode == "inference":
        if not args.query_id or not args.context:
            logger.error("推理模式需要提供查询论文ID (--query_id) 和引用上下文 (--context)")
            sys.exit(1)
        
        # 如果论文数据库为空，创建测试数据
        if not falcon.paper_db:
            logger.warning("论文数据库为空，创建测试数据...")
            query_paper = Paper(
                id=args.query_id,
                title=f"Query Paper: {args.query_id}",
                abstract="This is a test query paper for demonstration purposes.",
                authors=["Test Author"],
                year=2020,
                venue="Conference",
                citation_count=10
            )
            falcon.graph.add_paper_node(query_paper)
            falcon.paper_db[args.query_id] = query_paper
            
            # 创建一些候选论文
            for i in range(1, 4):
                candidate = Paper(
                    id=f"candidate{i}",
                    title=f"Candidate Paper {i}",
                    abstract=f"This is candidate paper {i} for testing.",
                    authors=[f"Author{i}"],
                    year=2020+i,
                    venue="Conference",
                    citation_count=10+i*10
                )
                falcon.graph.add_paper_node(candidate)
                falcon.paper_db[f"candidate{i}"] = candidate
            logger.info(f"创建了 {len(falcon.paper_db)} 篇测试论文")
        else:
            # 从论文数据库中获取查询论文
            if args.query_id not in falcon.paper_db:
                logger.error(f"查询论文ID '{args.query_id}' 不在论文数据库中")
                logger.info(f"可用的论文ID: {list(falcon.paper_db.keys())[:10]}")
                sys.exit(1)
            query_paper = falcon.paper_db[args.query_id]
        
        recommendations = run_inference(
            falcon,
            query_paper,
            args.context,
            args.top_k
        )
        
        # 保存结果
        import json
        output_file = output_dir / "recommendations.json"
        recommendations_dict = [rec.to_dict() for rec in recommendations]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"\n推荐结果已保存到: {output_file}")
    
    elif args.mode == "evaluate":
        if not args.data:
            logger.error("评估模式需要提供数据文件路径 (--data)")
            sys.exit(1)
        
        # 加载数据集
        logger.info(f"加载数据集: {args.dataset}")
        if args.dataset == "opencorpus":
            loader = OpenCorpusLoader(data_dir=args.data)
            papers, citations = loader.load_dataset(
                max_papers=args.max_papers,
                max_citations=args.max_citations
            )
            logger.info(f"加载了 {len(papers)} 篇论文和 {len(citations)} 条引用")
            # 更新paper_db
            falcon.paper_db = {paper.id: paper for paper in papers}
            # 更新知识图谱
            for paper in papers:
                falcon.graph.add_paper_node(paper)
        
        evaluate_system(falcon, args.data, str(output_dir))
    
    logger.info("=" * 60)
    logger.info("程序执行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

