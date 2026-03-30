"""完整工作流主脚本"""
import argparse
import sys
from pathlib import Path
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


def run_step(step_name: str, script_path: str, args: list) -> bool:
    """运行单个步骤
    
    Args:
        step_name: 步骤名称
        script_path: 脚本路径
        args: 命令行参数列表
        
    Returns:
        是否成功
    """
    import subprocess
    
    logger.info("=" * 60)
    logger.info(f"Step: {step_name}")
    logger.info("=" * 60)
    
    cmd = ["python3", script_path] + args
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ {step_name} 完成")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} 失败")
        logger.error(f"错误输出: {e.stderr}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FALCON完整工作流")
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "all", "acquire", "preprocess", "extract", "build_graph",
            "embeddings", "train", "inference", "evaluate"
        ],
        default="all",
        help="要执行的步骤"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="检查点目录"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Semantic Scholar API密钥（可选）"
    )
    parser.add_argument(
        "--target_papers",
        type=int,
        default=70000,
        help="目标论文数量"
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="如果输出文件已存在则跳过"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    
    # 创建目录
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    scripts_dir = Path(__file__).parent
    
    success = True
    
    # Step 1: 数据获取
    if args.step in ["all", "acquire"]:
        papers_file = data_dir / "raw" / "papers" / "opencorpus_papers.json"
        if args.skip_if_exists and papers_file.exists():
            logger.info(f"跳过数据获取，文件已存在: {papers_file}")
        else:
            success = run_step(
                "数据获取",
                str(scripts_dir / "data_acquisition.py"),
                [
                    "--target_count", str(args.target_papers),
                    "--output", str(papers_file),
                    *(["--api_key", args.api_key] if args.api_key else [])
                ]
            )
            if not success:
                logger.error("数据获取失败，停止流程")
                return
    
    # Step 2: 数据预处理
    if args.step in ["all", "preprocess"]:
        processed_papers = data_dir / "processed" / "cleaned_papers.json"
        if args.skip_if_exists and processed_papers.exists():
            logger.info(f"跳过数据预处理，文件已存在: {processed_papers}")
        else:
            success = run_step(
                "数据预处理",
                str(scripts_dir / "data_preprocessing.py"),
                [
                    "--input", str(data_dir / "raw" / "papers" / "opencorpus_papers.json"),
                    "--output_dir", str(data_dir / "processed")
                ]
            )
            if not success:
                logger.error("数据预处理失败，停止流程")
                return
    
    # Step 3: 实体与关系抽取
    if args.step in ["all", "extract"]:
        entities_dir = data_dir / "processed" / "entities_relations" / "entities"
        if args.skip_if_exists and entities_dir.exists() and any(entities_dir.iterdir()):
            logger.info(f"跳过实体与关系抽取，目录已存在: {entities_dir}")
        else:
            success = run_step(
                "实体与关系抽取",
                str(scripts_dir / "entity_relation_extraction.py"),
                [
                    "--input", str(data_dir / "processed" / "cleaned_papers.json"),
                    "--output_dir", str(data_dir / "processed" / "entities_relations")
                ]
            )
            if not success:
                logger.error("实体与关系抽取失败，停止流程")
                return
    
    # Step 4: 知识图谱构建
    if args.step in ["all", "build_graph"]:
        graph_dir = data_dir / "knowledge_graph" / "knowledge_graph"
        if args.skip_if_exists and graph_dir.exists():
            logger.info(f"跳过知识图谱构建，目录已存在: {graph_dir}")
        else:
            success = run_step(
                "知识图谱构建",
                str(scripts_dir / "build_knowledge_graph.py"),
                [
                    "--papers", str(data_dir / "processed" / "cleaned_papers.json"),
                    "--entities_dir", str(data_dir / "processed" / "entities_relations" / "entities"),
                    "--relations_dir", str(data_dir / "processed" / "entities_relations" / "relations"),
                    "--output_dir", str(data_dir / "knowledge_graph")
                ]
            )
            if not success:
                logger.error("知识图谱构建失败，停止流程")
                return
    
    # Step 5: 嵌入计算
    if args.step in ["all", "embeddings"]:
        embeddings_file = data_dir / "processed" / "embeddings" / "paper_embeddings.npy"
        if args.skip_if_exists and embeddings_file.exists():
            logger.info(f"跳过嵌入计算，文件已存在: {embeddings_file}")
        else:
            success = run_step(
                "嵌入计算",
                str(scripts_dir / "compute_embeddings.py"),
                [
                    "--papers", str(data_dir / "processed" / "cleaned_papers.json"),
                    "--entities", str(data_dir / "processed" / "entities_relations" / "all_entities.json"),
                    "--graph", str(data_dir / "knowledge_graph" / "knowledge_graph"),
                    "--output_dir", str(data_dir / "processed" / "embeddings"),
                    "--build_index"
                ]
            )
            if not success:
                logger.error("嵌入计算失败，停止流程")
                return
    
    # Step 6: 模型训练
    if args.step in ["all", "train"]:
        # 检查检查点是否已存在
        if args.skip_if_exists:
            entity_checkpoint = checkpoints_dir / "entity_extractor" / "best.pt"
            if entity_checkpoint.exists():
                logger.info(f"跳过模型训练，检查点已存在")
            else:
                success = run_step(
                    "模型训练",
                    str(scripts_dir / "train_models.py"),
                    [
                        "--train_papers", str(data_dir / "splits" / "train_papers.json"),
                        "--train_citations", str(data_dir / "splits" / "train_citations.json"),
                        "--entities_dir", str(data_dir / "processed" / "entities_relations" / "entities"),
                        "--output_dir", str(checkpoints_dir),
                        "--model", "all"
                    ]
                )
                if not success:
                    logger.error("模型训练失败，停止流程")
                    return
        else:
            success = run_step(
                "模型训练",
                str(scripts_dir / "train_models.py"),
                [
                    "--train_papers", str(data_dir / "splits" / "train_papers.json"),
                    "--train_citations", str(data_dir / "splits" / "train_citations.json"),
                    "--entities_dir", str(data_dir / "processed" / "entities_relations" / "entities"),
                    "--output_dir", str(checkpoints_dir),
                    "--model", "all"
                ]
            )
            if not success:
                logger.error("模型训练失败，停止流程")
                return
    
    # Step 7: 推理
    if args.step in ["all", "inference"]:
        logger.info("=" * 60)
        logger.info("Step: 推理")
        logger.info("=" * 60)
        logger.info("推理步骤需要使用main.py的inference模式")
        logger.info("示例命令:")
        logger.info(f"  python main.py --mode inference \\")
        logger.info(f"    --dataset opencorpus \\")
        logger.info(f"    --data {data_dir} \\")
        logger.info(f"    --graph {data_dir / 'knowledge_graph' / 'knowledge_graph'} \\")
        logger.info(f"    --paper_db {data_dir / 'processed' / 'cleaned_papers.json'} \\")
        logger.info(f"    --query_id <query_paper_id> \\")
        logger.info(f"    --context \"<citation_context>\" \\")
        logger.info(f"    --top_k 20 \\")
        logger.info(f"    --output {output_dir}")
    
    # Step 8: 评估
    if args.step in ["all", "evaluate"]:
        logger.info("=" * 60)
        logger.info("Step: 评估")
        logger.info("=" * 60)
        logger.info("评估步骤需要使用main.py的evaluate模式")
        logger.info("示例命令:")
        logger.info(f"  python main.py --mode evaluate \\")
        logger.info(f"    --dataset opencorpus \\")
        logger.info(f"    --data {data_dir} \\")
        logger.info(f"    --graph {data_dir / 'knowledge_graph' / 'knowledge_graph'} \\")
        logger.info(f"    --paper_db {data_dir / 'processed' / 'cleaned_papers.json'} \\")
        logger.info(f"    --output {output_dir}")
    
    logger.info("=" * 60)
    logger.info("工作流执行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

