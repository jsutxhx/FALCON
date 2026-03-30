"""实体抽取器模块"""
from typing import Optional, List
from pathlib import Path
import math
from transformers import AutoTokenizer, BertForTokenClassification
import torch
from src.utils.device_utils import get_device, move_to_device
from src.knowledge_graph.bio_tags import NUM_LABELS, id_to_label, get_entity_type_from_label
from src.data_processing.data_structures import Entity


class EntityExtractor:
    """BIO序列标注实体抽取器
    
    功能：
    - 使用SciBERT模型进行命名实体识别
    - 从文本中抽取细粒度实体（Task, Method, Material, Metric, Other）
    
    Attributes:
        tokenizer: SciBERT tokenizer
        model: BERT模型（用于token分类）
        device: 计算设备
        max_length: 最大序列长度
        label_map: 标签ID到标签字符串的映射
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 512,
        device: Optional[str] = None,
        num_labels: int = NUM_LABELS
    ):
        """初始化实体抽取器
        
        Args:
            model_path: 微调后的模型检查点路径。如果为None，则使用预训练模型
            model_name: 预训练模型名称，默认为 "allenai/scibert_scivocab_uncased"
            max_length: 最大序列长度，默认为512
            device: 计算设备 ("cuda", "cpu" 或 None自动检测)
            num_labels: 标签数量，默认为NUM_LABELS（11）
        """
        # 获取设备
        self.device = get_device(device)
        self.max_length = max_length
        
        # 加载tokenizer（总是使用预训练模型的tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        if model_path is not None:
            # 从检查点加载微调后的模型
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
            self.model = BertForTokenClassification.from_pretrained(
                str(model_path_obj),
                num_labels=num_labels
            )
        else:
            # 使用预训练模型（需要指定num_labels）
            self.model = BertForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 创建标签映射（ID到标签字符串）
        self.label_map = {i: id_to_label(i) for i in range(num_labels)}
    
    def extract(self, text: str) -> List[Entity]:
        """从文本中抽取实体
        
        功能：
        - Tokenize输入文本
        - 模型预测BIO标签
        - 解码标签为实体列表
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        if not text or not text.strip():
            return []
        
        # Tokenize输入文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True  # 用于恢复原始文本位置
        )
        
        # 将输入移动到设备
        inputs = move_to_device(inputs, self.device)
        
        # 模型预测BIO标签
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 解码标签为实体列表
        entities = self._decode_bio(
            text,
            inputs['input_ids'],
            inputs.get('offset_mapping'),
            predictions
        )
        
        return entities
    
    def _decode_bio(
        self,
        text: str,
        input_ids: torch.Tensor,
        offset_mapping: Optional[torch.Tensor],
        predictions: torch.Tensor
    ) -> List[Entity]:
        """解码BIO标签为实体列表
        
        功能：
        - 处理B-I连续序列
        - 合并subword tokens
        - 恢复原始文本片段（使用offset_mapping）
        
        Args:
            text: 原始文本
            input_ids: token IDs
            offset_mapping: token到原始文本的偏移映射（可选）
            predictions: 预测的标签ID，形状为 (batch_size, seq_len)
            
        Returns:
            实体列表
        """
        # 获取第一个样本的预测（batch_size=1）
        label_ids = predictions[0].cpu().numpy()
        
        # 将token IDs转换为tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # 获取offset_mapping（如果可用）
        offsets = None
        if offset_mapping is not None:
            offsets = offset_mapping[0].cpu().numpy()
        
        # 解码实体
        entities = []
        current_entity = None
        entity_counter = 0
        
        for i, (token, label_id) in enumerate(zip(tokens, label_ids)):
            # 跳过特殊token（[CLS], [SEP], [PAD]等）
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.label_map[label_id]
            
            # 获取当前token的offset（如果可用）
            token_start = None
            token_end = None
            if offsets is not None and i < len(offsets):
                token_start, token_end = offsets[i]
                # offset_mapping中，特殊token的offset通常是(0, 0)
                if token_start == 0 and token_end == 0:
                    continue
            
            # 处理B标签（开始新实体）
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity is not None:
                    entities.append(current_entity)
                
                # 开始新实体
                entity_type = get_entity_type_from_label(label)
                if entity_type is not None:
                    entity_counter += 1
                    current_entity = {
                        'entity_type': entity_type.value,
                        'token_indices': [i],
                        'text_start': token_start,
                        'text_end': token_end
                    }
            
            # 处理I标签（继续当前实体）
            elif label.startswith('I-'):
                entity_type = get_entity_type_from_label(label)
                if current_entity is not None and current_entity['entity_type'] == entity_type.value:
                    # 继续当前实体（B-I连续序列）
                    current_entity['token_indices'].append(i)
                    # 更新文本结束位置
                    if token_end is not None:
                        current_entity['text_end'] = token_end
                else:
                    # I标签不匹配当前实体，结束当前实体
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = None
                    
                    # I标签不匹配时，可以将其视为B标签（处理不规范的标签序列）
                    if entity_type is not None:
                        entity_counter += 1
                        current_entity = {
                            'entity_type': entity_type.value,
                            'token_indices': [i],
                            'text_start': token_start,
                            'text_end': token_end
                        }
            
            # 处理O标签（结束当前实体）
            else:  # label == 'O'
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # 保存最后一个实体
        if current_entity is not None:
            entities.append(current_entity)
        
        # 将token序列转换为实体对象
        result_entities = []
        for entity_dict in entities:
            # 恢复原始文本片段
            if entity_dict['text_start'] is not None and entity_dict['text_end'] is not None:
                # 使用offset_mapping恢复原始文本
                entity_text = text[entity_dict['text_start']:entity_dict['text_end']]
            else:
                # 回退到合并tokens的方法
                entity_tokens = [tokens[idx] for idx in entity_dict['token_indices']]
                token_texts = []
                for token in entity_tokens:
                    # 处理BERT的subword标记（##前缀）
                    if token.startswith('##'):
                        token_texts.append(token[2:])
                    else:
                        # 在非subword token前添加空格（除了第一个）
                        if token_texts:
                            token_texts.append(' ')
                        token_texts.append(token)
                
                entity_text = ''.join(token_texts).replace('▁', ' ').strip()
            
            # 创建Entity对象
            entity = Entity(
                id=f"entity_{entity_counter}",
                text=entity_text,
                entity_type=entity_dict['entity_type'],
                canonical=entity_text.lower().strip(),  # 简单的规范化
                weight=1.0
            )
            result_entities.append(entity)
            entity_counter += 1
        
        return result_entities
    
    def _compute_entity_weight(
        self,
        entity: Entity,
        text: str,
        is_title: bool = False
    ) -> float:
        """计算实体权重
        
        功能：
        - 根据实体在标题中的出现、位置和频率计算权重
        - 公式: w = Normalize((I(title) + 1/(1+pos)) * log(1+freq))
        
        Args:
            entity: 实体对象
            text: 文本内容（标题或摘要）
            is_title: 是否为标题文本，默认为False
            
        Returns:
            归一化后的权重值（0-1之间）
        """
        # I(title): 指示函数，如果实体在标题中出现则为1，否则为0
        i_title = 1.0 if is_title else 0.0
        
        # 计算实体在文本中的位置（字符位置，归一化到0-1）
        entity_text_lower = entity.text.lower()
        text_lower = text.lower()
        
        # 查找实体第一次出现的位置
        pos_char = text_lower.find(entity_text_lower)
        if pos_char == -1:
            # 如果找不到，使用文本长度作为位置（位置很靠后）
            pos_char = len(text)
        
        # 位置归一化：pos = 字符位置 / 文本长度
        # 使用 1/(1+pos) 使得位置越靠前，值越大
        pos_normalized = pos_char / max(len(text), 1)  # 归一化到0-1
        pos_score = 1.0 / (1.0 + pos_normalized)
        
        # 计算实体在文本中出现的频率
        # 使用不区分大小写的匹配
        freq = text_lower.count(entity_text_lower)
        freq_score = math.log(1.0 + freq)
        
        # 计算原始权重: (I(title) + 1/(1+pos)) * log(1+freq)
        raw_weight = (i_title + pos_score) * freq_score
        
        # 归一化权重（使用简单的min-max归一化）
        # 注意：这里使用固定的归一化范围，实际应用中可能需要根据所有实体的权重范围进行归一化
        # 为了简化，我们使用sigmoid函数进行归一化
        normalized_weight = 1.0 / (1.0 + math.exp(-raw_weight))
        
        return normalized_weight

