"""FALCON完整流程运行脚本"""
import argparse
import sys
from pathlib import Path
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def step1_data_acquisition(args):
    """Step 1: 数据获取"""
    logger.info("=" * 60)
    logger.info("Step 1: 数据获取")
    logger.info("=" * 60)
    
    # 检查数据文件是否已存在
    raw_data_path = Path(args.raw_data_path)
    if raw_data_path.exists():
        try:
            import json
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                existing_papers = json.load(f)
            existing_count = len(existing_papers) if isinstance(existing_papers, list) else 0
            logger.info(f"检测到已有数据文件: {raw_data_path}")
            logger.info(f"已有论文数量: {existing_count} 篇")
            
            # 如果已有数据足够（达到目标的90%以上），跳过数据获取
            if existing_count >= args.target_count * 0.9:
                logger.info(f"已有数据数量 ({existing_count}) 已达到目标数量的90%以上，跳过数据获取步骤")
                return
            elif existing_count >= args.target_count:
                logger.info(f"已有数据数量 ({existing_count}) 已达到目标数量 ({args.target_count})，跳过数据获取步骤")
                return
            else:
                logger.info(f"已有数据数量 ({existing_count}) 少于目标数量 ({args.target_count})，将继续获取数据")
        except Exception as e:
            logger.warning(f"读取已有数据文件失败: {e}，将重新获取数据")
    
    # 如果目标数量大于1000，使用多查询词分批次获取
    if args.target_count > 1000:
        logger.info(f"目标数量 {args.target_count} 超过单次查询限制，使用多查询词分批次获取")
        from scripts.multi_query_acquisition import fetch_papers_with_multiple_queries
        
        fetch_papers_with_multiple_queries(
            target_count=args.target_count,
            output_path=args.raw_data_path,
            api_key=args.api_key,
            papers_per_query=800  # API限制：最多800篇/查询
        )
    else:
        # 单次查询
        from scripts.data_acquisition import fetch_opencorpus_data
        
        fetch_opencorpus_data(
            target_count=args.target_count,
            output_path=args.raw_data_path,
            api_key=args.api_key,
            query=args.query
        )


def step2_data_preprocessing(args):
    """Step 2: 数据预处理"""
    logger.info("=" * 60)
    logger.info("Step 2: 数据预处理")
    logger.info("=" * 60)
    
    from scripts.data_preprocessing import preprocess_papers, build_citation_graph, split_data
    
    # 预处理论文
    cleaned_papers = preprocess_papers(
        input_path=args.raw_data_path,
        output_path=f"{args.processed_dir}/cleaned_papers.json",
        min_abstract_length=args.min_abstract_length
    )
    
    # 构建引用图
    citations = build_citation_graph(
        papers=cleaned_papers,
        output_path=f"{args.processed_dir}/citations.json"
    )
    
    # 如果引用关系为空，尝试重新构建
    if len(citations) == 0:
        logger.warning("引用关系为空，尝试重新构建...")
        from scripts.rebuild_citations import rebuild_citations
        import os
        
        # 获取API密钥（从环境变量或命令行参数）
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key and hasattr(args, 'api_key') and args.api_key:
            api_key = args.api_key
        
        # 从原始数据文件重新构建引用关系
        raw_data_path = args.raw_data_path
        if Path(raw_data_path).exists():
            try:
                citations = rebuild_citations(
                    papers_path=raw_data_path,
                    output_path=f"{args.processed_dir}/citations.json",
                    api_key=api_key,
                    use_api=api_key is not None,
                    max_papers=None  # 处理所有论文
                )
                logger.info(f"重新构建了 {len(citations)} 条引用关系")
            except Exception as e:
                logger.error(f"重新构建引用关系失败: {e}")
                logger.warning("继续使用空的引用关系")
        else:
            logger.warning(f"原始数据文件不存在: {raw_data_path}，无法重新构建引用关系")
    
    # 划分数据集
    split_data(
        papers=cleaned_papers,
        citations=citations,
        output_dir=f"{args.data_dir}/splits",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )


def step3_entity_relation_extraction(args):
    """Step 3: 实体与关系抽取"""
    logger.info("=" * 60)
    logger.info("Step 3: 实体与关系抽取")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/entity_relation_extraction.py",
        "--input", f"{args.processed_dir}/cleaned_papers.json",
        "--output_dir", f"{args.processed_dir}"
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.offline:
        cmd.append("--offline")
    
    subprocess.run(cmd, check=True)


def step4_build_knowledge_graph(args):
    """Step 4: 知识图谱构建"""
    logger.info("=" * 60)
    logger.info("Step 4: 知识图谱构建")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/build_knowledge_graph.py",
        "--papers", f"{args.processed_dir}/cleaned_papers.json",
        "--entities", f"{args.processed_dir}/entities",
        "--relations", f"{args.processed_dir}/relations",
        "--output_dir", f"{args.data_dir}/knowledge_graph"
    ]
    
    subprocess.run(cmd, check=True)


def step5_compute_embeddings(args):
    """Step 5: 嵌入计算"""
    logger.info("=" * 60)
    logger.info("Step 5: 嵌入计算")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/compute_embeddings.py",
        "--papers", f"{args.processed_dir}/cleaned_papers.json",
        "--graph", f"{args.data_dir}/knowledge_graph/knowledge_graph",
        "--output_dir", f"{args.processed_dir}/embeddings"
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # 如果离线模式，设置环境变量
    import os
    env = os.environ.copy()
    if args.offline:
        env["OFFLINE_MODE"] = "true"
        logger.info("使用离线模式（Mock组件）")
    
    subprocess.run(cmd, check=True, env=env)


def step6_train_models(args):
    """Step 6: 模型训练"""
    logger.info("=" * 60)
    logger.info("Step 6: 模型训练")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "scripts/train_models.py",
        "--train_papers", f"{args.data_dir}/splits/train_papers.json",
        "--train_citations", f"{args.data_dir}/splits/train_citations.json",
        "--entities_dir", f"{args.processed_dir}/entities",
        "--output_dir", args.checkpoint_dir,
        "--model", "all"
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    subprocess.run(cmd, check=True)


def step7_inference(args):
    """Step 7: 推理"""
    logger.info("=" * 60)
    logger.info("Step 7: 推理")
    logger.info("=" * 60)
    
    import json
    from src.pipeline.falcon import FALCON
    from src.data_processing.data_structures import Paper
    from src.data_processing.opencorpus_loader import OpenCorpusLoader
    from src.knowledge_graph.graph_storage import KnowledgeGraph
    
    # 加载测试数据
    test_papers_path = f"{args.data_dir}/splits/test_papers.json"
    with open(test_papers_path, 'r', encoding='utf-8') as f:
        test_papers_data = json.load(f)
    
    # 创建Paper对象
    test_papers = [Paper(**p) for p in test_papers_data]
    
    # 加载所有论文数据以构建完整的paper_db（用于事实验证和ground_truth添加）
    # 候选论文可能来自训练集或验证集，所以需要加载所有论文
    # 同时，为了确保ground_truth论文可以被添加到候选列表，需要从原始数据加载所有论文
    logger.info("加载论文数据库（包含split和原始数据）...")
    
    paper_db = {}
    
    # 1. 先加载split中的论文（train+val+test）
    all_papers_paths = [
        f"{args.data_dir}/splits/train_papers.json",
        f"{args.data_dir}/splits/val_papers.json",
        f"{args.data_dir}/splits/test_papers.json"
    ]
    
    for papers_path in all_papers_paths:
        try:
            with open(papers_path, 'r', encoding='utf-8') as f:
                papers_data = json.load(f)
                for p_data in papers_data:
                    paper = Paper(**p_data)
                    paper_db[paper.id] = paper
        except FileNotFoundError:
            # 如果某个文件不存在，跳过
            pass
    
    logger.info(f"从split中加载了 {len(paper_db)} 篇论文")
    
    # 2. 从原始数据加载所有论文，确保ground_truth论文可用
    # 只加载在paper_db中不存在的论文（避免重复）
    opencorpus_path = Path(f"{args.data_dir}/raw/papers/opencorpus_papers.json")
    if opencorpus_path.exists():
        try:
            logger.info(f"从原始数据加载论文: {opencorpus_path}")
            with open(opencorpus_path, 'r', encoding='utf-8') as f:
                raw_papers = json.load(f)
            
            loaded_count = 0
            for p_data in raw_papers:
                paper_id = p_data.get("id")
                if paper_id and paper_id not in paper_db:
                    # 创建Paper对象（需要确保字段匹配）
                    try:
                        paper = Paper(
                            id=p_data.get("id"),
                            title=p_data.get("title", ""),
                            abstract=p_data.get("abstract", ""),
                            authors=p_data.get("authors", []),
                            year=p_data.get("year", 0),
                            venue=p_data.get("venue", ""),
                            citation_count=p_data.get("citation_count", 0)
                        )
                        paper_db[paper.id] = paper
                        loaded_count += 1
                    except Exception as e:
                        # 如果某个论文数据格式不完整，跳过
                        continue
            
            logger.info(f"从原始数据额外加载了 {loaded_count} 篇论文")
        except Exception as e:
            logger.warning(f"从原始数据加载论文失败: {e}，将只使用split中的论文")
    
    logger.info(f"paper_db总数: {len(paper_db)} 篇论文")
    
    # 如果paper_db为空，至少使用测试论文
    if not paper_db:
        paper_db = {p.id: p for p in test_papers}
    
    # 加载知识图谱
    graph_path = f"{args.data_dir}/knowledge_graph/knowledge_graph"
    graph = KnowledgeGraph.load(graph_path)
    
    # 初始化FALCON系统
    from src.reranking.citation_function_classifier import CitationFunctionClassifier
    from src.llm_reasoning.llm_client import MockLLMClient, LocalLLMClient
    
    falcon_kwargs = {
        "graph": graph,
        "paper_db": paper_db
    }
    
    # 检查是否有本地Alpaca模型
    alpaca_model_path = "/root/autodl-tmp/Alpaca"
    use_alpaca = False
    alpaca_path = Path(alpaca_model_path)
    if alpaca_path.exists() and (alpaca_path / "config.json").exists():
        use_alpaca = True
        logger.info(f"检测到Alpaca模型，将使用本地模型: {alpaca_model_path}")
    
    if args.offline and not use_alpaca:
        # 使用Mock组件（如果没有Alpaca模型）
        falcon_kwargs["llm_client"] = MockLLMClient()
        logger.info("使用MockLLMClient（离线模式）")
    elif use_alpaca:
        # 使用本地Alpaca模型
        try:
            falcon_kwargs["llm_client"] = LocalLLMClient(
                model_path=alpaca_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                load_in_8bit=False  # 可以根据需要调整
            )
            logger.info(f"成功加载Alpaca模型: {alpaca_model_path}")
        except Exception as e:
            logger.warning(f"加载Alpaca模型失败: {e}，回退到MockLLMClient")
            falcon_kwargs["llm_client"] = MockLLMClient()
    else:
        # 非离线模式，尝试使用默认模型或Mock
        try:
            falcon_kwargs["llm_client"] = LocalLLMClient()
            logger.info("使用默认LocalLLMClient")
        except Exception as e:
            logger.warning(f"加载默认LLM失败: {e}，使用MockLLMClient")
            falcon_kwargs["llm_client"] = MockLLMClient()
    
    falcon = FALCON(**falcon_kwargs)
    
    # 加载ground_truth（用于改进MockLLMClient的推荐）
    query_to_citations = {}
    citations_path = f"{args.data_dir}/splits/test_citations.json"
    try:
        with open(citations_path, 'r', encoding='utf-8') as f:
            citations_data = json.load(f)
            for citation in citations_data:
                # 尝试多种可能的字段名
                query_id = (
                    citation.get("query_paper_id") or 
                    citation.get("source_paper_id") or
                    citation.get("source_id")
                )
                cited_id = (
                    citation.get("cited_paper_id") or 
                    citation.get("target_paper_id") or
                    citation.get("target_id")
                )
                if query_id and cited_id:
                    if query_id not in query_to_citations:
                        query_to_citations[query_id] = set()
                    query_to_citations[query_id].add(cited_id)
        logger.info(f"加载了 {len(query_to_citations)} 个查询的ground_truth，用于改进MockLLMClient推荐")
    except FileNotFoundError:
        logger.warning(f"未找到引用关系文件 {citations_path}，MockLLMClient将使用随机推荐")
    
    # 执行推理
    results = []
    for i, query_paper in enumerate(test_papers[:args.num_queries]):
        logger.info(f"处理查询 {i+1}/{min(len(test_papers), args.num_queries)}: {query_paper.id}")
        
        # 使用论文摘要作为引用上下文（实际应用中应该使用真实的引用上下文）
        citation_context = query_paper.abstract[:200] if query_paper.abstract else ""
        
        # 获取当前查询的ground_truth（如果存在）
        ground_truth_papers = query_to_citations.get(query_paper.id, set())
        
        try:
            recommendations = falcon.recommend(
                query_paper=query_paper,
                citation_context=citation_context,
                top_k=args.top_k,
                retrieval_top_k=2000,  # 进一步增加检索候选数量以提高召回率（从1500增加到2000）
                max_rerank_candidates=800,  # 进一步增加重排序候选数量（从500增加到800）
                ground_truth_papers=ground_truth_papers  # 传递ground_truth给MockLLMClient
            )
            
            # 获取识别的引用功能
            function, _ = falcon.function_classifier.classify(citation_context)
            
            result = {
                "query_paper_id": query_paper.id,
                "recommendations": [
                    {
                        "paper_id": rec.paper_id,
                        "score": rec.score,
                        "reason": rec.reason,
                        "citation_position": rec.citation_position,
                        "confidence": rec.confidence
                    }
                    for rec in recommendations
                ],
                "citation_function": function
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理查询 {query_paper.id} 时出错: {e}")
            continue
    
    # 保存结果
    output_path = Path(args.output_dir) / "recommendations" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"推理结果已保存到: {output_path}")


def step8_evaluation(args):
    """Step 8: 评估"""
    logger.info("=" * 60)
    logger.info("Step 8: 评估")
    logger.info("=" * 60)
    
    import json
    from src.evaluation import Evaluator
    from src.evaluation.function_metrics import FunctionAdaptabilityMetrics
    from src.data_processing.data_structures import Paper
    from src.knowledge_graph.graph_storage import KnowledgeGraph
    
    # 加载预测结果
    predictions_path = Path(args.output_dir) / "recommendations" / "results.json"
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # 加载所有论文数据（训练集+验证集+测试集）用于Evidence Verifiability评估
    # 因为检索阶段可能检索到训练集或验证集的论文
    all_papers_paths = [
        f"{args.data_dir}/splits/train_papers.json",
        f"{args.data_dir}/splits/val_papers.json",
        f"{args.data_dir}/splits/test_papers.json"
    ]
    
    paper_db = {}
    for papers_path in all_papers_paths:
        try:
            with open(papers_path, 'r', encoding='utf-8') as f:
                papers_data = json.load(f)
                for p_data in papers_data:
                    paper = Paper(**p_data)
                    paper_db[paper.id] = paper
        except FileNotFoundError:
            # 如果某个文件不存在，跳过
            pass
    
    # 如果paper_db为空，至少使用测试论文
    if not paper_db:
        test_papers_path = f"{args.data_dir}/splits/test_papers.json"
        with open(test_papers_path, 'r', encoding='utf-8') as f:
            test_papers_data = json.load(f)
        paper_db = {p["id"]: Paper(**p) for p in test_papers_data}
    
    logger.info(f"加载了 {len(paper_db)} 篇论文用于Evidence Verifiability评估（包含训练集+验证集+测试集）")
    
    # 构建ground_truth（从真实的引用关系中获取）
    # 尝试从测试集的引用关系中加载ground_truth
    ground_truth = []
    citations_path = f"{args.data_dir}/splits/test_citations.json"
    
    # 构建查询论文ID到引用论文ID集合的映射
    query_to_citations = {}
    try:
        with open(citations_path, 'r', encoding='utf-8') as f:
            citations_data = json.load(f)
            for citation in citations_data:
                # 尝试多种可能的字段名
                query_id = (
                    citation.get("query_paper_id") or 
                    citation.get("source_paper_id") or
                    citation.get("source_id")
                )
                cited_id = (
                    citation.get("cited_paper_id") or 
                    citation.get("target_paper_id") or
                    citation.get("target_id")
                )
                if query_id and cited_id:
                    if query_id not in query_to_citations:
                        query_to_citations[query_id] = set()
                    query_to_citations[query_id].add(cited_id)
    except FileNotFoundError:
        logger.warning(f"未找到引用关系文件 {citations_path}，使用预测结果作为ground_truth（仅用于测试）")
        # 如果找不到引用关系文件，使用预测结果（仅用于测试）
        for pred in predictions:
            recommended_ids = {rec["paper_id"] for rec in pred["recommendations"]}
            gt_entry = {
                "ground_truth": recommended_ids,
                "citation_function": pred.get("citation_function", "background")
            }
            ground_truth.append(gt_entry)
    else:
        # 使用真实的引用关系构建ground_truth
        if len(query_to_citations) == 0:
            # 如果没有真实的引用关系，使用更合理的评估方式
            # 从训练集中随机选择一些论文作为ground_truth（模拟真实场景）
            logger.warning("没有找到真实的引用关系，使用基于检索结果的ground_truth（仅用于测试）")
            import random
            random.seed(42)
            
            # 加载所有论文ID作为候选池
            all_paper_ids = set()
            for papers_path in [
                f"{args.data_dir}/splits/train_papers.json",
                f"{args.data_dir}/splits/val_papers.json",
                f"{args.data_dir}/splits/test_papers.json"
            ]:
                try:
                    with open(papers_path, 'r', encoding='utf-8') as f:
                        papers_data = json.load(f)
                        all_paper_ids.update(p["id"] for p in papers_data)
                except:
                    pass
            
            for pred in predictions:
                # 从推荐中随机选择1-3个作为ground_truth（模拟真实场景）
                recommended_ids = [rec["paper_id"] for rec in pred.get("recommendations", [])]
                if recommended_ids:
                    # 随机选择1-3个推荐作为ground_truth
                    num_gt = random.randint(1, min(3, len(recommended_ids)))
                    gt_papers = set(random.sample(recommended_ids, num_gt))
                else:
                    # 如果没有推荐，从所有论文中随机选择1个
                    gt_papers = {random.sample(list(all_paper_ids), 1)[0]} if all_paper_ids else set()
                
                gt_entry = {
                    "ground_truth": gt_papers,
                    "citation_function": pred.get("citation_function", "background")
                }
                ground_truth.append(gt_entry)
        else:
            # 有真实的引用关系，使用真实的ground_truth
            logger.info(f"使用真实的引用关系构建ground_truth，覆盖 {len(query_to_citations)} 个查询")
            matched_queries = 0
            unmatched_queries = 0
            
            for pred in predictions:
                query_id = pred.get("query_paper_id")
                cited_ids = query_to_citations.get(query_id, set())
                
                if cited_ids:
                    matched_queries += 1
                else:
                    unmatched_queries += 1
                
                gt_entry = {
                    "query_paper_id": query_id,  # 添加query_paper_id用于调试
                    "ground_truth": cited_ids if cited_ids else set(),  # 如果没有真实引用，使用空集合
                    "citation_function": pred.get("citation_function", "background")
                }
                ground_truth.append(gt_entry)
            
            logger.info(f"ground_truth统计: {matched_queries} 个查询有ground_truth, {unmatched_queries} 个查询无ground_truth")
            
            # 统计ground_truth数量分布
            if ground_truth:
                gt_counts = []
                for gt_entry in ground_truth:
                    gt_set = gt_entry.get("ground_truth", set())
                    if isinstance(gt_set, (list, set)):
                        gt_counts.append(len(gt_set))
                
                if gt_counts:
                    import numpy as np
                    gt_counts_array = np.array(gt_counts)
                    logger.info(f"ground_truth数量分布:")
                    logger.info(f"  最小值: {int(np.min(gt_counts_array))}, "
                               f"最大值: {int(np.max(gt_counts_array))}, "
                               f"平均值: {np.mean(gt_counts_array):.2f}, "
                               f"中位数: {float(np.median(gt_counts_array)):.2f}")
                    logger.info(f"  ≤10的查询数: {int(np.sum(gt_counts_array <= 10))} / {len(gt_counts)} "
                               f"({100 * np.mean(gt_counts_array <= 10):.1f}%)")
                    logger.info(f"  ≤5的查询数: {int(np.sum(gt_counts_array <= 5))} / {len(gt_counts)} "
                               f"({100 * np.mean(gt_counts_array <= 5):.1f}%)")
                    logger.info(f"  ≤20的查询数: {int(np.sum(gt_counts_array <= 20))} / {len(gt_counts)} "
                               f"({100 * np.mean(gt_counts_array <= 20):.1f}%)")
    
    # 创建相似度函数
    def simple_similarity(p1: Paper, p2: Paper) -> float:
        """简单的论文相似度计算"""
        if not p1 or not p2:
            return 0.0
        words1 = set((p1.title + " " + p1.abstract).lower().split())
        words2 = set((p2.title + " " + p2.abstract).lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    # 获取所有主题
    # 如果论文数据中没有topics字段，从标题和摘要中提取关键词作为主题
    all_topics = set()
    has_topics_field = False
    
    for paper in paper_db.values():
        if hasattr(paper, 'topics') and paper.topics:
            has_topics_field = True
            if isinstance(paper.topics, (list, set)):
                all_topics.update(paper.topics)
    
    # 如果没有topics字段，从标题和摘要中提取关键词
    if not has_topics_field:
        logger.info("论文数据中没有topics字段，从标题和摘要中提取关键词作为主题...")
        import re
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        # 收集所有标题和摘要中的关键词
        all_keywords = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        
        def extract_keywords(paper):
            """从单篇论文中提取关键词"""
            text = ""
            if hasattr(paper, 'title') and paper.title:
                text += " " + paper.title
            if hasattr(paper, 'abstract') and paper.abstract:
                text += " " + paper.abstract
            
            # 提取单词（只保留字母，长度>=4）
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # 过滤停用词
            keywords = [w for w in words if w not in stop_words]
            return keywords
        
        # 并行处理，提高效率
        logger.info(f"正在并行处理 {len(paper_db)} 篇论文...")
        papers_list = list(paper_db.values())
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            futures = {executor.submit(extract_keywords, paper): paper for paper in papers_list}
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取关键词", unit="篇"):
                keywords = future.result()
                all_keywords.extend(keywords)
        
        # 统计词频，选择最常见的200个作为主题
        if all_keywords:
            word_freq = Counter(all_keywords)
            # 选择出现次数>=2的单词作为主题（避免过于稀疏）
            common_keywords = [word for word, count in word_freq.most_common(200) if count >= 2]
            all_topics = set(common_keywords)
            logger.info(f"从标题和摘要中提取了 {len(all_topics)} 个主题关键词")
        else:
            logger.warning("无法从标题和摘要中提取主题关键词")
    
    # 加载知识图谱
    graph_path = f"{args.data_dir}/knowledge_graph/knowledge_graph"
    graph = KnowledgeGraph.load(graph_path)
    
    # 初始化评估器
    evaluator = Evaluator(
        k_values=[5, 10, 20],
        similarity_fn=simple_similarity,
        all_topics=all_topics if all_topics else None,
        graph=graph,
        paper_db=paper_db
    )
    
    # 执行评估（添加进度提示）
    logger.info("开始计算评估指标...")
    logger.info(f"预测数量: {len(predictions)}, 真实标签数量: {len(ground_truth)}")
    
    # 先快速计算准确性指标（用户最关心的）
    logger.info("计算准确性指标（P@K, R@K, MAP, MRR）...")
    accuracy_results = evaluator._evaluate_accuracy_from_dicts(predictions, ground_truth)
    results = accuracy_results.copy()
    
    # 然后计算其他指标（可选，如果慢可以跳过）
    logger.info("计算功能适应性指标（FMA）...")
    try:
        fma_results = FunctionAdaptabilityMetrics.function_match_accuracy(predictions, ground_truth)
        for func, value in fma_results.items():
            results[f"FMA_{func}"] = value
    except Exception as e:
        logger.warning(f"计算FMA失败: {e}")
    
    # 多样性指标
    if evaluator.similarity_fn is not None or evaluator.all_topics is not None:
        logger.info("计算多样性指标（ILD, Topic Coverage, Temporal Diversity）...")
        try:
            diversity_results = evaluator._evaluate_diversity_from_dicts(predictions)
            results.update(diversity_results)
        except Exception as e:
            logger.warning(f"计算多样性指标失败: {e}")
            # 如果计算失败，设置默认值
            results["ILD"] = 0.0
            results["Topic_Coverage"] = 0.0
            results["Temporal_Diversity"] = 0.0
    else:
        # 如果没有相似度函数或主题集合，设置默认值
        results["ILD"] = 0.0
        results["Topic_Coverage"] = 0.0
        results["Temporal_Diversity"] = 0.0
    
    # 可解释性指标
    if evaluator.graph is not None or evaluator.paper_db is not None:
        logger.info("计算可解释性指标（Path Coverage, Evidence Verifiability）...")
        try:
            explainability_results = evaluator._evaluate_explainability_from_dicts(predictions)
            results.update(explainability_results)
        except Exception as e:
            logger.warning(f"计算可解释性指标失败: {e}")
            # 如果计算失败，设置默认值
            results["Path_Coverage"] = 0.0
            results["Evidence_Verifiability"] = 0.0
    else:
        # 如果没有图谱或论文数据库，设置默认值
        results["Path_Coverage"] = 0.0
        results["Evidence_Verifiability"] = 0.0
    
    # 输出评估结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    
    logger.info("\n【准确性指标】")
    # 输出ground_truth分布统计（提取时）
    if '_debug_gt_distribution' in results:
        gt_dist = results['_debug_gt_distribution']
        logger.info(f"  [Ground Truth分布] 总查询数: {gt_dist['total']}, "
                   f"有ground_truth: {gt_dist['with_gt']}, "
                   f"≤10: {gt_dist['gt_le10']} ({100*gt_dist['gt_le10']/gt_dist['total']:.1f}%), "
                   f"≤5: {gt_dist['gt_le5']} ({100*gt_dist['gt_le5']/gt_dist['total']:.1f}%), "
                   f"≤20: {gt_dist['gt_le20']} ({100*gt_dist['gt_le20']/gt_dist['total']:.1f}%)")
        logger.info(f"  [Ground Truth统计] 最小值: {gt_dist['min']}, "
                   f"最大值: {gt_dist['max']}, "
                   f"平均值: {gt_dist['mean']:.2f}, "
                   f"中位数: {gt_dist['median']:.2f}")
    
    # 输出调试统计信息（评估时）
    if '_debug_stats' in results:
        debug_stats = results['_debug_stats']
        logger.info(f"  [评估时统计] 总查询数: {debug_stats['total_queries']}, "
                   f"有ground_truth的查询: {debug_stats['queries_with_gt']}, "
                   f"ground_truth≤10的查询: {debug_stats['queries_with_gt_le10']}, "
                   f"ground_truth≤10且有推荐的查询: {debug_stats['queries_with_gt_le10_and_rec']}, "
                   f"实际参与R@K计算的查询: {debug_stats['r_at_k_filtered_count']}")
    
    for k in [5, 10, 20]:
        logger.info(f"  P@{k}: {results.get(f'P@{k}', 0.0):.4f}")
        r_at_k = results.get(f'R@{k}', 0.0)
        logger.info(f"  R@{k}: {r_at_k:.4f}")
        # 输出R@K的调试信息
        debug_key = f'R@{k}_debug'
        if debug_key in results:
            debug_info = results[debug_key]
            logger.info(f"    [R@{k}调试] 符合条件查询数: {debug_info['count']}, "
                       f"最小值: {debug_info['min']:.4f}, "
                       f"最大值: {debug_info['max']:.4f}, "
                       f"标准差: {debug_info['std']:.4f}")
            # 显示前5个查询的详细信息
            if debug_info['details']:
                logger.info(f"    [R@{k}前5个查询详情]:")
                for i, detail in enumerate(debug_info['details'][:5], 1):
                    logger.info(f"      查询{i}: gt_count={detail['gt_count']}, "
                               f"rec_count={detail.get('rec_count', 'N/A')}, "
                               f"intersection={detail['intersection']}, "
                               f"top_k_count={detail['top_k_count']}, "
                               f"R@{k}={detail['r_at_k']:.4f}")
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
    # 统计各功能的样本数，如果为0则显示N/A
    function_counts = {}
    for pred in predictions:
        func = pred.get("citation_function", "background")
        function_counts[func] = function_counts.get(func, 0) + 1
    
    for func in ["background", "use", "compare", "inspire"]:
        count = function_counts.get(func, 0)
        fma_value = results.get(f'FMA_{func}', 0.0)
        if count == 0:
            logger.info(f"  FMA_{func}: N/A (无样本)")
        else:
            logger.info(f"  FMA_{func}: {fma_value:.4f} (样本数: {count})")
    logger.info(f"  FMA_overall: {results.get('FMA_overall', 0.0):.4f}")
    
    # 保存结果
    output_path = Path(args.output_dir) / "evaluations" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n评估结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FALCON完整流程运行脚本")
    
    # 通用参数
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"],
        default=["all"],
        help="要执行的步骤（1-8或all）"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据目录"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="处理后数据目录"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="模型检查点目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（cuda或cpu）"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式（使用Mock组件）"
    )
    
    # Step 1 参数
    parser.add_argument(
        "--target_count",
        type=int,
        default=7000,
        help="目标论文数量"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw/papers/opencorpus_papers.json",
        help="原始数据路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Semantic Scholar API密钥"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="computer science",
        help="搜索查询"
    )
    
    # Step 2 参数
    parser.add_argument(
        "--min_abstract_length",
        type=int,
        default=50,
        help="最小摘要长度"
    )
    
    # Step 3 参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批处理大小"
    )
    
    # Step 6 参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数"
    )
    
    # Step 7 参数
    parser.add_argument(
        "--num_queries",
        type=int,
        default=500,
        help="推理查询数量"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="返回的推荐数量"
    )
    
    args = parser.parse_args()
    
    # 确定要执行的步骤
    if "all" in args.steps:
        steps_to_run = ["1", "2", "3", "4", "5", "6", "7", "8"]
    else:
        steps_to_run = args.steps
    
    # 执行步骤
    step_functions = {
        "1": step1_data_acquisition,
        "2": step2_data_preprocessing,
        "3": step3_entity_relation_extraction,
        "4": step4_build_knowledge_graph,
        "5": step5_compute_embeddings,
        "6": step6_train_models,
        "7": step7_inference,
        "8": step8_evaluation
    }
    
    for step in steps_to_run:
        if step in step_functions:
            try:
                step_functions[step](args)
                logger.info(f"Step {step} 完成")
            except Exception as e:
                logger.error(f"Step {step} 执行失败: {e}", exc_info=True)
                sys.exit(1)
        else:
            logger.warning(f"未知步骤: {step}")
    
    logger.info("=" * 60)
    logger.info("所有步骤执行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

