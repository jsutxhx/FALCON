"""关系抽取器模块"""
from typing import Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.utils.device_utils import get_device, move_to_device
from src.knowledge_graph.schema import RelationType
from src.data_processing.data_structures import Entity, Relation


class RelationExtractor:
    """关系抽取器
    
    功能：
    - 使用SciBERT模型进行关系分类
    - 判断两个实体之间的关系类型（CONTAINS, HIERARCHY, IMPLEMENT, USE, EVALUATE）
    
    Attributes:
        tokenizer: SciBERT tokenizer
        model: BERT模型（用于序列分类）
        device: 计算设备
        max_length: 最大序列长度
        num_relations: 关系类型数量（5）
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 256,
        device: Optional[str] = None,
        num_relations: int = 5
    ):
        """初始化关系抽取器
        
        Args:
            model_path: 微调后的模型检查点路径。如果为None，则使用预训练模型
            model_name: 预训练模型名称，默认为 "allenai/scibert_scivocab_uncased"
            max_length: 最大序列长度，默认为256
            device: 计算设备 ("cuda", "cpu" 或 None自动检测)
            num_relations: 关系类型数量，默认为5
        """
        # 获取设备
        self.device = get_device(device)
        self.max_length = max_length
        self.num_relations = num_relations
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        if model_path is not None:
            # 从检查点加载微调后的模型
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path_obj),
                num_labels=num_relations
            )
        else:
            # 使用预训练模型（需要指定num_labels）
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_relations
            )
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 创建关系类型映射（ID到RelationType）
        self.id_to_relation = {
            i: list(RelationType)[i] for i in range(num_relations)
        }
    
    def extract(
        self,
        entity1: Entity,
        entity2: Entity,
        context: str,
        threshold: float = 0.5
    ) -> Optional[Relation]:
        """抽取两个实体之间的关系
        
        功能：
        - 构造输入: [CLS] e1 [SEP] context [SEP] e2
        - 使用模型分类预测关系类型
        - 返回Relation对象或None（如果置信度低于阈值）
        
        Args:
            entity1: 头实体
            entity2: 尾实体
            context: 上下文文本（包含两个实体的句子或段落）
            threshold: 置信度阈值，低于此值返回None，默认为0.5
            
        Returns:
            Relation对象，如果置信度低于阈值则返回None
        """
        # 构造输入文本: [CLS] e1 [SEP] context [SEP] e2
        # 注意：tokenizer会自动添加[CLS]和[SEP]
        input_text = f"{entity1.text} [SEP] {context} [SEP] {entity2.text}"
        
        # Tokenize输入文本
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # 将输入移动到设备
        inputs = move_to_device(inputs, self.device)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1)[0]
            
            # 获取预测的关系类型ID
            predicted_id = torch.argmax(probs).item()
            confidence = probs[predicted_id].item()
        
        # 如果置信度低于阈值，返回None
        if confidence < threshold:
            return None
        
        # 获取关系类型
        relation_type_enum = self.id_to_relation[predicted_id]
        relation_type = relation_type_enum.value
        
        # 创建Relation对象
        relation = Relation(
            head_id=entity1.id,
            tail_id=entity2.id,
            relation_type=relation_type,
            weight=confidence
        )
        
        return relation

