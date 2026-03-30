"""图谱构建器模块"""
from typing import List, Optional, Dict
from tqdm import tqdm
from src.data_processing.data_structures import Paper, Entity
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.entity_linker import EntityLinker
from src.knowledge_graph.relation_extractor import RelationExtractor
from src.knowledge_graph.graph_storage import KnowledgeGraph


class GraphBuilder:
    """知识图谱构建器
    
    功能：
    - 整合实体抽取、实体链接、关系抽取和图谱存储
    - 从论文数据构建完整知识图谱
    
    Attributes:
        entity_extractor: 实体抽取器
        entity_linker: 实体链接器
        relation_extractor: 关系抽取器
    """
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        entity_linker: Optional[EntityLinker] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        device: Optional[str] = None
    ):
        """初始化图谱构建器
        
        Args:
            entity_extractor: 实体抽取器实例。如果为None，则创建新的实例
            entity_linker: 实体链接器实例。如果为None，则创建新的实例
            relation_extractor: 关系抽取器实例。如果为None，则创建新的实例
            device: 计算设备（仅在创建新实例时使用）
        """
        # 初始化组件
        if entity_extractor is None:
            self.entity_extractor = EntityExtractor(device=device)
        else:
            self.entity_extractor = entity_extractor
        
        if entity_linker is None:
            self.entity_linker = EntityLinker()
        else:
            self.entity_linker = entity_linker
        
        if relation_extractor is None:
            self.relation_extractor = RelationExtractor(device=device)
        else:
            self.relation_extractor = relation_extractor
    
    def build(
        self,
        papers: List[Paper],
        extract_relations: bool = True,
        relation_threshold: float = 0.5,
        show_progress: bool = True
    ) -> KnowledgeGraph:
        """从论文数据构建知识图谱
        
        功能：
        1. 对每篇论文：
           - 添加论文节点
           - 从标题和摘要中抽取实体
           - 规范化实体并链接
           - 添加实体节点（如果不存在）
           - 添加contains边（论文->实体）
        2. 对实体对：
           - 抽取关系（使用上下文）
           - 如果关系存在，添加关系边
        
        Args:
            papers: 论文列表
            extract_relations: 是否抽取实体间关系，默认为True
            relation_threshold: 关系抽取的置信度阈值，默认为0.5
            show_progress: 是否显示进度条，默认为True
            
        Returns:
            构建完成的知识图谱
        """
        # 创建知识图谱
        kg = KnowledgeGraph()
        
        # 用于实体链接的字典：canonical -> entity_id
        canonical_to_entity: Dict[str, str] = {}
        # 用于存储每篇论文的实体列表：paper_id -> List[Entity]
        paper_entities: Dict[str, List[Entity]] = {}
        # 实体计数器（用于生成唯一ID）
        entity_counter = 0
        
        # 第一步：处理论文和实体抽取
        iterator = tqdm(papers, desc="处理论文") if show_progress else papers
        for paper in iterator:
            # 添加论文节点
            kg.add_paper_node(paper)
            
            # 从标题和摘要中抽取实体
            text = f"{paper.title} {paper.abstract}"
            entities = self.entity_extractor.extract(text)
            
            # 存储论文的实体列表
            paper_entities[paper.id] = []
            
            # 处理每个实体
            for entity in entities:
                # 规范化实体文本
                canonical = self.entity_linker.normalize(entity.text)
                
                # 实体链接：如果规范化形式已存在，使用已有实体ID
                if canonical in canonical_to_entity:
                    # 实体已存在，使用已有ID
                    entity_id = canonical_to_entity[canonical]
                    # 更新实体权重（取较大值）
                    existing_entity = kg.get_node(entity_id)
                    if existing_entity:
                        existing_weight = existing_entity.get("weight", 1.0)
                        if entity.weight > existing_weight:
                            # 更新权重（需要重新添加节点）
                            entity.id = entity_id
                            entity.canonical = canonical
                            kg.add_entity_node(entity)
                else:
                    # 新实体，创建新ID
                    entity_counter += 1
                    entity_id = f"entity_{entity_counter}"
                    entity.id = entity_id
                    entity.canonical = canonical
                    
                    # 添加到图谱
                    kg.add_entity_node(entity)
                    canonical_to_entity[canonical] = entity_id
                
                # 添加contains边（论文->实体）
                kg.add_contains_edge(paper.id, entity_id, weight=entity.weight)
                
                # 保存实体引用（用于后续关系抽取）
                # 创建新的Entity对象保存引用（保持原始entity不变）
                entity_ref = Entity(
                    id=entity_id,
                    text=entity.text,
                    entity_type=entity.entity_type,
                    canonical=canonical,
                    embedding=entity.embedding,
                    weight=entity.weight
                )
                paper_entities[paper.id].append(entity_ref)
        
        # 第二步：抽取实体间关系
        if extract_relations:
            # 收集所有实体对（在同一篇论文中的实体对）
            entity_pairs = []
            for paper_id, entities in paper_entities.items():
                if len(entities) < 2:
                    continue
                
                # 获取论文文本作为上下文
                paper = next(p for p in papers if p.id == paper_id)
                context = f"{paper.title} {paper.abstract}"
                
                # 生成实体对
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        entity_pairs.append((entities[i], entities[j], context))
            
            # 抽取关系
            if entity_pairs:
                iterator = tqdm(entity_pairs, desc="抽取关系") if show_progress else entity_pairs
                for entity1, entity2, context in iterator:
                    # 抽取关系
                    relation = self.relation_extractor.extract(
                        entity1, entity2, context, threshold=relation_threshold
                    )
                    
                    if relation is not None:
                        # 添加关系边（跳过contains关系，因为已经添加了）
                        if relation.relation_type != "contains":
                            kg.add_relation_edge(
                                relation.head_id,
                                relation.tail_id,
                                relation.relation_type,
                                weight=relation.weight
                            )
        
        return kg
