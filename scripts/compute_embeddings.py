"""嵌入计算脚本"""
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import faiss

# 添加项目根目录到路径，以便导入src模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processing.data_structures import Paper, Entity
from src.embeddings.specter_encoder import SpecterEncoder
from src.embeddings.entity_embedder import EntityEmbedder
from src.knowledge_graph.graph_storage import KnowledgeGraph

# 尝试导入RelationEmbedder，如果不存在则使用Mock
try:
    from src.embeddings.relation_embedder import RelationEmbedder
except ImportError:
    logger.warning("RelationEmbedder模块不存在，将使用Mock实现")
    RelationEmbedder = None


def compute_paper_embeddings(
    papers: List[Paper],
    output_path: str,
    encoder: SpecterEncoder = None
) -> np.ndarray:
    """计算论文嵌入
    
    Args:
        papers: 论文列表
        output_path: 输出文件路径
        encoder: SPECTER编码器（如果为None，会创建默认实例）
        
    Returns:
        论文嵌入矩阵 (n_papers, 768)
    """
    logger.info(f"开始计算论文嵌入，论文数量: {len(papers)}")
    
    if encoder is None:
        # 检查是否离线模式
        offline_mode = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        if offline_mode:
            logger.warning("使用Mock SPECTER编码器（功能有限）")
            # 创建Mock编码器
            class MockSpecterEncoder:
                def encode_paper(self, title: str, abstract: str):
                    # 返回随机向量作为Mock
                    import numpy as np
                    return np.random.randn(768).astype(np.float32)
                
                def encode_papers(self, papers):
                    import numpy as np
                    return np.random.randn(len(papers), 768).astype(np.float32)
                
                def encode(self, paper):
                    # 兼容encode方法（接受Paper对象）
                    return self.encode_paper(paper.title, paper.abstract)
            encoder = MockSpecterEncoder()
        else:
            try:
                logger.info("初始化SPECTER编码器...")
                encoder = SpecterEncoder()
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning(f"无法加载SPECTER编码器（网络问题），使用Mock模式: {e}")
                class MockSpecterEncoder:
                    def encode_paper(self, title: str, abstract: str):
                        import numpy as np
                        return np.random.randn(768).astype(np.float32)
                    
                    def encode_papers(self, papers):
                        import numpy as np
                        return np.random.randn(len(papers), 768).astype(np.float32)
                    
                    def encode(self, title: str, abstract: str):
                        # 兼容encode方法
                        return self.encode_paper(title, abstract)
                encoder = MockSpecterEncoder()
    
    embeddings = []
    paper_ids = []
    
    for i, paper in enumerate(papers):
        if i % 100 == 0:
            logger.info(f"处理论文 {i}/{len(papers)}")
        
        try:
            # 使用encode_paper方法，传入title和abstract
            embedding = encoder.encode_paper(paper.title, paper.abstract)
            embeddings.append(embedding)
            paper_ids.append(paper.id)
        except Exception as e:
            logger.warning(f"计算论文 {paper.id} 的嵌入时出错: {e}")
            # 使用零向量作为占位符
            embeddings.append(np.zeros(768))
            paper_ids.append(paper.id)
    
    embeddings_array = np.array(embeddings)
    
    # 保存嵌入
    logger.info(f"保存论文嵌入到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_array)
    
    # 保存论文ID映射
    id_mapping_path = output_path.parent / f"{output_path.stem}_ids.json"
    with open(id_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(paper_ids, f, ensure_ascii=False, indent=2)
    
    logger.info(f"论文嵌入计算完成，形状: {embeddings_array.shape}")
    return embeddings_array


def compute_entity_embeddings(
    entities: List[Entity],
    output_path: str,
    embedder: EntityEmbedder = None
) -> np.ndarray:
    """计算实体嵌入
    
    Args:
        entities: 实体列表
        output_path: 输出文件路径
        embedder: 实体嵌入器（如果为None，会创建默认实例）
        
    Returns:
        实体嵌入矩阵 (n_entities, 768)
    """
    logger.info(f"开始计算实体嵌入，实体数量: {len(entities)}")
    
    if embedder is None:
        logger.info("初始化实体嵌入器...")
        embedder = EntityEmbedder()
    
    embeddings = []
    entity_ids = []
    
    for i, entity in enumerate(entities):
        if i % 1000 == 0:
            logger.info(f"处理实体 {i}/{len(entities)}")
        
        try:
            embedding = embedder.embed(entity)
            embeddings.append(embedding)
            entity_ids.append(entity.id)
        except Exception as e:
            logger.warning(f"计算实体 {entity.id} 的嵌入时出错: {e}")
            embeddings.append(np.zeros(768))
            entity_ids.append(entity.id)
    
    embeddings_array = np.array(embeddings)
    
    # 保存嵌入
    logger.info(f"保存实体嵌入到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_array)
    
    # 保存实体ID映射
    id_mapping_path = output_path.parent / f"{output_path.stem}_ids.json"
    with open(id_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(entity_ids, f, ensure_ascii=False, indent=2)
    
    logger.info(f"实体嵌入计算完成，形状: {embeddings_array.shape}")
    return embeddings_array


def compute_relation_embeddings(
    graph: KnowledgeGraph,
    output_path: str,
    embedder: RelationEmbedder = None,
    dimension: int = 128
) -> np.ndarray:
    """计算关系嵌入（使用TransE）
    
    Args:
        graph: 知识图谱
        output_path: 输出文件路径
        embedder: 关系嵌入器（如果为None，会创建默认实例）
        dimension: 嵌入维度
        
    Returns:
        关系嵌入矩阵
    """
    logger.info("开始计算关系嵌入")
    
    if embedder is None:
        # 检查是否离线模式
        offline_mode = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        if RelationEmbedder is None or offline_mode:
            # 使用Mock实现
            logger.warning("使用Mock关系嵌入器（功能有限）")
            class MockRelationEmbedder:
                def __init__(self, dimension=128):
                    self.dimension = dimension
                
                def train(self, graph):
                    logger.info("Mock关系嵌入器：跳过训练")
                    pass
                
                def get_relation_types(self):
                    return []
                
                def get_relation_embedding(self, rel_type):
                    return np.zeros(self.dimension)
                
                def get_relation_embeddings(self):
                    return np.zeros((0, self.dimension))
            
            embedder = MockRelationEmbedder(dimension=dimension)
        else:
            try:
                logger.info("初始化关系嵌入器（TransE）...")
                embedder = RelationEmbedder(dimension=dimension)
            except (OSError, ConnectionError, TimeoutError) as e:
                logger.warning(f"无法加载关系嵌入器（网络问题），使用Mock模式: {e}")
                class MockRelationEmbedder:
                    def __init__(self, dimension=128):
                        self.dimension = dimension
                    
                    def train(self, graph):
                        logger.info("Mock关系嵌入器：跳过训练")
                        pass
                    
                    def get_relation_types(self):
                        return []
                    
                    def get_relation_embedding(self, rel_type):
                        return np.zeros(self.dimension)
                    
                    def get_relation_embeddings(self):
                        return np.zeros((0, self.dimension))
                embedder = MockRelationEmbedder(dimension=dimension)
    
    # 训练TransE模型
    logger.info("训练TransE模型...")
    embedder.train(graph)
    
    # 获取关系嵌入
    relation_types = embedder.get_relation_types()
    embeddings = []
    relation_names = []
    
    for rel_type in relation_types:
        embedding = embedder.get_relation_embedding(rel_type)
        embeddings.append(embedding)
        relation_names.append(rel_type)
    
    embeddings_array = np.array(embeddings)
    
    # 保存嵌入
    logger.info(f"保存关系嵌入到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_array)
    
    # 保存关系名称映射
    name_mapping_path = output_path.parent / f"{output_path.stem}_names.json"
    with open(name_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(relation_names, f, ensure_ascii=False, indent=2)
    
    logger.info(f"关系嵌入计算完成，形状: {embeddings_array.shape}")
    return embeddings_array


def build_faiss_index(
    embeddings: np.ndarray,
    output_path: str,
    index_type: str = "L2"
) -> None:
    """构建FAISS索引
    
    Args:
        embeddings: 嵌入矩阵
        output_path: 输出文件路径
        index_type: 索引类型（"L2" 或 "IP"）
    """
    logger.info(f"构建FAISS索引，嵌入形状: {embeddings.shape}")
    
    dimension = embeddings.shape[1]
    
    # 创建索引
    if index_type == "L2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IP":
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
    
    # 归一化嵌入（对于IP索引）
    if index_type == "IP":
        faiss.normalize_L2(embeddings)
    
    # 添加向量
    index.add(embeddings.astype('float32'))
    
    # 保存索引
    logger.info(f"保存FAISS索引到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    
    logger.info(f"FAISS索引构建完成，包含 {index.ntotal} 个向量")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="计算嵌入")
    parser.add_argument(
        "--papers",
        type=str,
        required=True,
        help="论文JSON文件路径"
    )
    parser.add_argument(
        "--entities",
        type=str,
        default=None,
        help="实体JSON文件路径（可选）"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="知识图谱路径（可选，用于关系嵌入）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/embeddings",
        help="输出目录"
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="是否构建FAISS索引"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 计算论文嵌入
    paper_embeddings = compute_paper_embeddings(
        papers=papers,
        output_path=str(output_dir / "paper_embeddings.npy")
    )
    
    # 构建论文FAISS索引
    if args.build_index:
        build_faiss_index(
            embeddings=paper_embeddings,
            output_path=str(output_dir / "paper_faiss.index"),
            index_type="L2"
        )
    
    # 计算实体嵌入（如果提供）
    if args.entities:
        logger.info(f"加载实体数据: {args.entities}")
        with open(args.entities, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        
        entities = []
        for e_data in entities_data:
            if isinstance(e_data, dict):
                entities.append(Entity(**e_data))
            else:
                entities.append(e_data)
        
        entity_embeddings = compute_entity_embeddings(
            entities=entities,
            output_path=str(output_dir / "entity_embeddings.npy")
        )
        
        # 构建实体FAISS索引
        if args.build_index:
            build_faiss_index(
                embeddings=entity_embeddings,
                output_path=str(output_dir / "entity_faiss.index"),
                index_type="L2"
            )
    
    # 计算关系嵌入（如果提供知识图谱）
    if args.graph:
        logger.info(f"加载知识图谱: {args.graph}")
        graph = KnowledgeGraph.load(args.graph)
        
        relation_embeddings = compute_relation_embeddings(
            graph=graph,
            output_path=str(output_dir / "relation_embeddings.npy")
        )
    
    logger.info("嵌入计算完成")


if __name__ == "__main__":
    main()

