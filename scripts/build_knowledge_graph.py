"""知识图谱构建脚本"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# 添加项目根目录到路径，以便导入src模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processing.data_structures import Paper, Entity, Relation
from src.knowledge_graph.graph_storage import KnowledgeGraph


def build_knowledge_graph(
    papers: List[Paper],
    entities_dir: str,
    relations_dir: str,
    output_dir: str
) -> KnowledgeGraph:
    """构建知识图谱
    
    Args:
        papers: 论文列表
        entities_dir: 实体文件目录
        relations_dir: 关系文件目录
        output_dir: 输出目录
        
    Returns:
        构建好的知识图谱
    """
    logger.info("开始构建知识图谱")
    
    graph = KnowledgeGraph()
    entities_dir = Path(entities_dir)
    relations_dir = Path(relations_dir)
    
    # Step 1: 添加论文节点
    logger.info(f"添加 {len(papers)} 个论文节点")
    for paper in papers:
        graph.add_paper_node(paper)
    
    # Step 2: 添加实体节点和Contains边
    logger.info("添加实体节点和Contains边")
    entity_count = 0
    contains_count = 0
    
    for paper in papers:
        entities_file = entities_dir / f"{paper.id}.json"
        if not entities_file.exists():
            continue
        
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            
            for entity_data in entities_data:
                # 创建实体对象
                entity = Entity(
                    id=entity_data.get("id", ""),
                    text=entity_data.get("text", ""),
                    entity_type=entity_data.get("entity_type", "other"),
                    canonical=entity_data.get("canonical", entity_data.get("text", "")),
                    weight=entity_data.get("weight", 1.0)
                )
                
                # 添加实体节点
                graph.add_entity_node(entity)
                entity_count += 1
                
                # 添加Contains边
                graph.add_contains_edge(paper.id, entity.id, weight=entity.weight)
                contains_count += 1
        
        except Exception as e:
            logger.warning(f"处理论文 {paper.id} 的实体时出错: {e}")
            continue
    
    logger.info(f"添加了 {entity_count} 个实体节点和 {contains_count} 条Contains边")
    
    # Step 3: 添加关系边
    logger.info("添加关系边")
    relation_count = 0
    
    for paper in papers:
        relations_file = relations_dir / f"{paper.id}.json"
        if not relations_file.exists():
            continue
        
        try:
            with open(relations_file, 'r', encoding='utf-8') as f:
                relations_data = json.load(f)
            
            for relation_data in relations_data:
                source_id = relation_data.get("source_entity_id", "")
                target_id = relation_data.get("target_entity_id", "")
                relation_type = relation_data.get("relation_type", "")
                
                if source_id and target_id and relation_type:
                    graph.add_relation_edge(
                        source_id, target_id, relation_type,
                        weight=relation_data.get("weight", 1.0)
                    )
                    relation_count += 1
        
        except Exception as e:
            logger.warning(f"处理论文 {paper.id} 的关系时出错: {e}")
            continue
    
    logger.info(f"添加了 {relation_count} 条关系边")
    
    # Step 4: 保存知识图谱
    logger.info("保存知识图谱")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = output_dir / "knowledge_graph"
    graph.save(str(graph_path))
    
    logger.info(f"知识图谱已保存到: {graph_path}")
    logger.info(f"图谱统计: {graph.num_nodes()} 个节点, {graph.num_edges()} 条边")
    
    return graph


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建知识图谱")
    parser.add_argument(
        "--papers",
        type=str,
        required=True,
        help="论文JSON文件路径"
    )
    parser.add_argument(
        "--entities_dir",
        type=str,
        required=True,
        help="实体文件目录"
    )
    parser.add_argument(
        "--relations_dir",
        type=str,
        required=True,
        help="关系文件目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/knowledge_graph",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 加载论文数据
    logger.info(f"加载论文数据: {args.papers}")
    with open(args.papers, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    papers = []
    for p_data in papers_data:
        if isinstance(p_data, dict):
            papers.append(Paper(**p_data))
        else:
            papers.append(p_data)
    
    # 构建知识图谱
    graph = build_knowledge_graph(
        papers=papers,
        entities_dir=args.entities_dir,
        relations_dir=args.relations_dir,
        output_dir=args.output_dir
    )
    
    logger.info("知识图谱构建完成")


if __name__ == "__main__":
    main()

