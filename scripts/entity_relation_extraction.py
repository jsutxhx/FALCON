"""实体与关系抽取脚本"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from loguru import logger

# 添加项目根目录到路径，以便导入src模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processing.data_structures import Paper, Entity, Relation
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.relation_extractor import RelationExtractor


def extract_entities_and_relations(
    papers: List[Paper],
    output_dir: str,
    entity_extractor: EntityExtractor = None,
    relation_extractor: RelationExtractor = None
) -> None:
    """抽取实体和关系
    
    Args:
        papers: 论文列表
        output_dir: 输出目录
        entity_extractor: 实体抽取器（如果为None，会创建默认实例）
        relation_extractor: 关系抽取器（如果为None，会创建默认实例）
    """
    logger.info(f"开始抽取实体和关系，论文数量: {len(papers)}")
    
    # 初始化抽取器
    if entity_extractor is None:
        logger.info("初始化实体抽取器...")
        entity_extractor = EntityExtractor()
    
    if relation_extractor is None:
        logger.info("初始化关系抽取器...")
        relation_extractor = RelationExtractor()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entities_dir = output_dir / "entities"
    relations_dir = output_dir / "relations"
    entities_dir.mkdir(parents=True, exist_ok=True)
    relations_dir.mkdir(parents=True, exist_ok=True)
    
    all_entities = []
    all_relations = []
    
    for i, paper in enumerate(papers):
        if i % 100 == 0:
            logger.info(f"处理论文 {i}/{len(papers)}")
        
        try:
            # 抽取实体
            text = f"{paper.title}. {paper.abstract}"
            entities = entity_extractor.extract(text)
            
            # 保存每篇论文的实体
            entities_file = entities_dir / f"{paper.id}.json"
            # 使用asdict转换Entity对象（dataclass）
            entities_dict = []
            for entity in entities:
                entity_dict = asdict(entity)
                # 处理numpy数组（如果有embedding）
                if 'embedding' in entity_dict and entity_dict['embedding'] is not None:
                    import numpy as np
                    if isinstance(entity_dict['embedding'], np.ndarray):
                        entity_dict['embedding'] = entity_dict['embedding'].tolist()
                entities_dict.append(entity_dict)
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_dict, f, ensure_ascii=False, indent=2)
            
            all_entities.extend(entities)
            
            # 抽取关系（实体对之间的关系）
            if len(entities) >= 2:
                relations = []
                for j in range(len(entities)):
                    for k in range(j + 1, len(entities)):
                        entity1 = entities[j]
                        entity2 = entities[k]
                        
                        # 抽取关系
                        relation = relation_extractor.extract(
                            entity1=entity1,
                            entity2=entity2,
                            context=text
                        )
                        
                        if relation:
                            relations.append(relation)
                            all_relations.append(relation)
                
                # 保存每篇论文的关系
                if relations:
                    relations_file = relations_dir / f"{paper.id}.json"
                    # 使用asdict转换Relation对象（dataclass）
                    relations_dict = [asdict(rel) for rel in relations]
                    with open(relations_file, 'w', encoding='utf-8') as f:
                        json.dump(relations_dict, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logger.warning(f"处理论文 {paper.id} 时出错: {e}")
            continue
    
    # 保存汇总数据
    logger.info(f"抽取了 {len(all_entities)} 个实体和 {len(all_relations)} 条关系")
    
    # 保存所有实体
    all_entities_file = output_dir / "all_entities.json"
    # 使用asdict转换Entity对象（dataclass）
    entities_dict = []
    for entity in all_entities:
        entity_dict = asdict(entity)
        # 处理numpy数组（如果有embedding）
        if 'embedding' in entity_dict and entity_dict['embedding'] is not None:
            import numpy as np
            if isinstance(entity_dict['embedding'], np.ndarray):
                entity_dict['embedding'] = entity_dict['embedding'].tolist()
        entities_dict.append(entity_dict)
    with open(all_entities_file, 'w', encoding='utf-8') as f:
        json.dump(entities_dict, f, ensure_ascii=False, indent=2)
    
    # 保存所有关系
    all_relations_file = output_dir / "all_relations.json"
    # 使用asdict转换Relation对象（dataclass）
    relations_dict = [asdict(rel) for rel in all_relations]
    with open(all_relations_file, 'w', encoding='utf-8') as f:
        json.dump(relations_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"实体和关系已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实体与关系抽取")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入论文JSON文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/entities_relations",
        help="输出目录"
    )
    parser.add_argument(
        "--entity_model",
        type=str,
        default=None,
        help="实体抽取模型路径（可选）"
    )
    parser.add_argument(
        "--relation_model",
        type=str,
        default=None,
        help="关系抽取模型路径（可选）"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式（使用Mock组件，避免网络连接）"
    )
    
    args = parser.parse_args()
    
    # 加载论文数据
    logger.info(f"加载论文数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    papers = [Paper.from_dict(p) if hasattr(Paper, 'from_dict') else Paper(**p) for p in papers_data]
    
    # 初始化抽取器
    entity_extractor = None
    relation_extractor = None
    
    # 检查是否离线模式（通过环境变量或参数）
    offline_mode = args.offline or os.getenv("OFFLINE_MODE", "false").lower() == "true"
    
    if offline_mode:
        logger.warning("使用离线模式（Mock组件），功能有限")
        # 创建Mock抽取器
        class MockEntityExtractor:
            def extract(self, text: str):
                # 简单的关键词匹配作为Mock
                entities = []
                keywords = {
                    "method": ["algorithm", "approach", "method", "technique", "model", "system"],
                    "task": ["task", "problem", "challenge", "goal", "objective"],
                    "material": ["dataset", "data", "corpus", "benchmark"],
                    "metric": ["accuracy", "precision", "recall", "F1", "score", "performance"]
                }
                text_lower = text.lower()
                entity_id = 0
                for entity_type, kw_list in keywords.items():
                    for kw in kw_list:
                        if kw in text_lower:
                            entities.append(Entity(
                                id=f"mock_{entity_type}_{entity_id}",
                                text=kw,
                                entity_type=entity_type,
                                canonical=kw,
                                weight=1.0
                            ))
                            entity_id += 1
                            if len(entities) >= 10:  # 限制数量
                                break
                    if len(entities) >= 10:
                        break
                return entities
        
        class MockRelationExtractor:
            def extract(self, head_entity, tail_entity, context: str):
                # 简单的Mock关系
                return [Relation(
                    head_id=head_entity.id,
                    tail_id=tail_entity.id,
                    relation_type="related",
                    weight=0.5
                )]
        
        entity_extractor = MockEntityExtractor()
        relation_extractor = MockRelationExtractor()
    else:
        # 尝试加载真实模型
        try:
            if args.entity_model:
                entity_extractor = EntityExtractor(model_path=args.entity_model)
            else:
                entity_extractor = EntityExtractor()
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"无法加载实体抽取器（网络问题），使用Mock模式: {e}")
            # 创建Mock抽取器
            class MockEntityExtractor:
                def extract(self, text: str):
                    entities = []
                    keywords = {
                        "method": ["algorithm", "approach", "method"],
                        "task": ["task", "problem"],
                        "material": ["dataset", "data"],
                        "metric": ["accuracy", "precision"]
                    }
                    text_lower = text.lower()
                    entity_id = 0
                    for entity_type, kw_list in keywords.items():
                        for kw in kw_list:
                            if kw in text_lower:
                                entities.append(Entity(
                                    id=f"mock_{entity_type}_{entity_id}",
                                    text=kw,
                                    entity_type=entity_type,
                                    canonical=kw,
                                    weight=1.0
                                ))
                                entity_id += 1
                                if len(entities) >= 10:
                                    break
                        if len(entities) >= 10:
                            break
                    return entities
            entity_extractor = MockEntityExtractor()
        
        try:
            if args.relation_model:
                relation_extractor = RelationExtractor(model_path=args.relation_model)
            else:
                relation_extractor = RelationExtractor()
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"无法加载关系抽取器（网络问题），使用Mock模式: {e}")
            class MockRelationExtractor:
                def extract(self, head_entity, tail_entity, context: str):
                    return [Relation(
                        head_id=head_entity.id,
                        tail_id=tail_entity.id,
                        relation_type="related",
                        weight=0.5
                    )]
            relation_extractor = MockRelationExtractor()
    
    # 抽取实体和关系
    extract_entities_and_relations(
        papers=papers,
        output_dir=args.output_dir,
        entity_extractor=entity_extractor,
        relation_extractor=relation_extractor
    )
    
    logger.info("实体与关系抽取完成")


if __name__ == "__main__":
    main()

