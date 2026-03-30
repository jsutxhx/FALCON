"""知识图谱存储模块"""
import networkx as nx
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from src.data_processing.data_structures import Paper, Entity, Relation
from src.utils.file_utils import save_json, load_json, ensure_dir


@dataclass
class Edge:
    """边数据类
    
    Attributes:
        source: 源节点ID
        target: 目标节点ID
        relation_type: 关系类型
        weight: 边权重
    """
    source: str
    target: str
    relation_type: str
    weight: float = 1.0


class KnowledgeGraph:
    """知识图谱存储类
    
    功能：
    - 使用networkx.DiGraph存储知识图谱
    - 管理论文节点和实体节点
    - 管理节点间的关系边
    
    Attributes:
        graph: networkx有向图对象
    """
    
    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.DiGraph()
    
    def add_paper_node(self, paper: Paper):
        """添加论文节点
        
        功能：
        - 将Paper对象添加到图中作为节点
        - 节点ID为paper.id
        - 节点属性包含Paper的所有字段
        
        Args:
            paper: Paper对象
        """
        # 将Paper对象转换为字典，存储为节点属性
        node_attrs = {
            "node_type": "paper",
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "doi": paper.doi
        }
        
        # 添加节点（如果节点已存在，会更新属性）
        self.graph.add_node(paper.id, **node_attrs)
    
    def add_entity_node(self, entity: Entity):
        """添加实体节点
        
        功能：
        - 将Entity对象添加到图中作为节点
        - 节点ID为entity.id
        - 节点属性包含Entity的所有字段（embedding需要特殊处理）
        
        Args:
            entity: Entity对象
        """
        # 将Entity对象转换为字典，存储为节点属性
        # 注意：embedding是numpy数组，需要转换为列表或单独存储
        node_attrs = {
            "node_type": "entity",
            "id": entity.id,
            "text": entity.text,
            "entity_type": entity.entity_type,
            "canonical": entity.canonical,
            "weight": entity.weight
        }
        
        # 如果embedding存在，转换为列表存储（networkx不支持numpy数组直接存储）
        if entity.embedding is not None:
            node_attrs["embedding"] = entity.embedding.tolist()
            node_attrs["embedding_shape"] = entity.embedding.shape
        else:
            node_attrs["embedding"] = None
            node_attrs["embedding_shape"] = None
        
        # 添加节点（如果节点已存在，会更新属性）
        self.graph.add_node(entity.id, **node_attrs)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点属性
        
        功能：
        - 通过节点ID查询节点属性
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点属性字典，如果节点不存在则返回None
        """
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
    
    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在
        
        Args:
            node_id: 节点ID
            
        Returns:
            如果节点存在返回True，否则返回False
        """
        return node_id in self.graph
    
    def num_nodes(self) -> int:
        """获取节点数量
        
        Returns:
            图中节点数量
        """
        return self.graph.number_of_nodes()
    
    def get_paper_nodes(self) -> List[str]:
        """获取所有论文节点ID
        
        Returns:
            论文节点ID列表
        """
        paper_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "paper":
                paper_nodes.append(node_id)
        return paper_nodes
    
    def get_entity_nodes(self) -> List[str]:
        """获取所有实体节点ID
        
        Returns:
            实体节点ID列表
        """
        entity_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") == "entity":
                entity_nodes.append(node_id)
        return entity_nodes
    
    def add_contains_edge(self, paper_id: str, entity_id: str, weight: float = 1.0):
        """添加论文包含实体的边
        
        功能：
        - 在论文节点和实体节点之间添加"contains"关系边
        - 边从paper_id指向entity_id
        
        Args:
            paper_id: 论文节点ID
            entity_id: 实体节点ID
            weight: 边权重，默认为1.0
        """
        # 验证节点存在（可选，networkx会自动创建不存在的节点）
        # 但为了数据一致性，我们要求节点必须存在
        if paper_id not in self.graph:
            raise ValueError(f"Paper node '{paper_id}' does not exist")
        if entity_id not in self.graph:
            raise ValueError(f"Entity node '{entity_id}' does not exist")
        
        # 添加边，边属性包含关系类型和权重
        edge_attrs = {
            "relation_type": "contains",
            "weight": weight
        }
        
        self.graph.add_edge(paper_id, entity_id, **edge_attrs)
    
    def add_relation_edge(
        self,
        entity1_id: str,
        entity2_id: str,
        relation_type: str,
        weight: float = 1.0
    ):
        """添加实体间关系边
        
        功能：
        - 在两个实体节点之间添加关系边
        - 边从entity1_id指向entity2_id
        - 关系类型可以是: hierarchy, implement, use, evaluate
        
        Args:
            entity1_id: 头实体节点ID
            entity2_id: 尾实体节点ID
            relation_type: 关系类型 (hierarchy, implement, use, evaluate)
            weight: 边权重，默认为1.0
        """
        # 验证关系类型
        allowed_relations = {"hierarchy", "implement", "use", "evaluate"}
        if relation_type not in allowed_relations:
            raise ValueError(
                f"relation_type must be one of {allowed_relations}, "
                f"got '{relation_type}'"
            )
        
        # 验证节点存在
        if entity1_id not in self.graph:
            raise ValueError(f"Entity node '{entity1_id}' does not exist")
        if entity2_id not in self.graph:
            raise ValueError(f"Entity node '{entity2_id}' does not exist")
        
        # 添加边，边属性包含关系类型和权重
        edge_attrs = {
            "relation_type": relation_type,
            "weight": weight
        }
        
        self.graph.add_edge(entity1_id, entity2_id, **edge_attrs)
    
    def get_edges(self, node_id: str) -> List[Edge]:
        """获取节点的所有边
        
        功能：
        - 获取指定节点的所有出边和入边
        - 返回Edge对象列表
        
        Args:
            node_id: 节点ID
            
        Returns:
            Edge对象列表，包含该节点的所有边（出边和入边）
        """
        if node_id not in self.graph:
            return []
        
        edges = []
        
        # 获取出边（从该节点出发的边）
        for target_id, edge_attrs in self.graph[node_id].items():
            relation_type = edge_attrs.get("relation_type", "unknown")
            weight = edge_attrs.get("weight", 1.0)
            edge = Edge(
                source=node_id,
                target=target_id,
                relation_type=relation_type,
                weight=weight
            )
            edges.append(edge)
        
        # 获取入边（指向该节点的边）
        for source_id, target_dict in self.graph.pred[node_id].items():
            edge_attrs = target_dict
            relation_type = edge_attrs.get("relation_type", "unknown")
            weight = edge_attrs.get("weight", 1.0)
            edge = Edge(
                source=source_id,
                target=node_id,
                relation_type=relation_type,
                weight=weight
            )
            edges.append(edge)
        
        return edges
    
    def num_edges(self) -> int:
        """获取边数量
        
        Returns:
            图中边的数量
        """
        return self.graph.number_of_edges()
    
    def get_out_edges(self, node_id: str) -> List[Edge]:
        """获取节点的所有出边
        
        Args:
            node_id: 节点ID
            
        Returns:
            Edge对象列表，包含该节点的所有出边
        """
        if node_id not in self.graph:
            return []
        
        edges = []
        for target_id, edge_attrs in self.graph[node_id].items():
            relation_type = edge_attrs.get("relation_type", "unknown")
            weight = edge_attrs.get("weight", 1.0)
            edge = Edge(
                source=node_id,
                target=target_id,
                relation_type=relation_type,
                weight=weight
            )
            edges.append(edge)
        
        return edges
    
    def get_in_edges(self, node_id: str) -> List[Edge]:
        """获取节点的所有入边
        
        Args:
            node_id: 节点ID
            
        Returns:
            Edge对象列表，包含该节点的所有入边
        """
        if node_id not in self.graph:
            return []
        
        edges = []
        for source_id, target_dict in self.graph.pred[node_id].items():
            edge_attrs = target_dict
            relation_type = edge_attrs.get("relation_type", "unknown")
            weight = edge_attrs.get("weight", 1.0)
            edge = Edge(
                source=source_id,
                target=node_id,
                relation_type=relation_type,
                weight=weight
            )
            edges.append(edge)
        
        return edges
    
    def save(self, path: str):
        """保存知识图谱到文件
        
        功能：
        - 将节点和边数据保存为JSON格式
        - 分别存储 nodes.json 和 edges.json
        
        Args:
            path: 保存路径（目录），会在该目录下创建 nodes.json 和 edges.json
        """
        path_obj = Path(path)
        ensure_dir(str(path_obj))
        
        # 序列化节点数据
        nodes_data = []
        for node_id, attrs in self.graph.nodes(data=True):
            # 复制节点属性，确保所有数据可序列化
            node_data = {
                "id": node_id,
                **attrs
            }
            nodes_data.append(node_data)
        
        # 序列化边数据
        edges_data = []
        for source_id, target_id, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source_id,
                "target": target_id,
                "relation_type": attrs.get("relation_type", "unknown"),
                "weight": attrs.get("weight", 1.0)
            }
            edges_data.append(edge_data)
        
        # 保存到文件
        nodes_file = path_obj / "nodes.json"
        edges_file = path_obj / "edges.json"
        
        save_json(nodes_data, str(nodes_file))
        save_json(edges_data, str(edges_file))
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """从文件加载知识图谱
        
        功能：
        - 从 nodes.json 和 edges.json 加载数据
        - 重建知识图谱结构
        
        Args:
            path: 加载路径（目录），应该包含 nodes.json 和 edges.json
            
        Returns:
            重建的KnowledgeGraph对象
        """
        path_obj = Path(path)
        
        # 加载节点和边数据
        nodes_file = path_obj / "nodes.json"
        edges_file = path_obj / "edges.json"
        
        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
        if not edges_file.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_file}")
        
        nodes_data = load_json(str(nodes_file))
        edges_data = load_json(str(edges_file))
        
        # 创建新的KnowledgeGraph实例
        kg = cls()
        
        # 重建节点
        for node_data in nodes_data:
            node_id = node_data.pop("id")
            # 恢复节点属性
            kg.graph.add_node(node_id, **node_data)
        
        # 重建边
        for edge_data in edges_data:
            source = edge_data["source"]
            target = edge_data["target"]
            relation_type = edge_data.get("relation_type", "unknown")
            weight = edge_data.get("weight", 1.0)
            
            edge_attrs = {
                "relation_type": relation_type,
                "weight": weight
            }
            kg.graph.add_edge(source, target, **edge_attrs)
        
        return kg

