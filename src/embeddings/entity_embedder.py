"""实体嵌入器模块"""
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from src.embeddings.specter_encoder import SpecterEncoder
from src.data_processing.data_structures import Entity


class EntityEmbedder:
    """实体嵌入器
    
    功能：
    - 使用SPECTER模型对实体文本进行编码
    - 支持缓存已编码的实体以提高效率
    
    Attributes:
        encoder: SPECTER编码器
        cache: 实体嵌入缓存字典，键为实体ID，值为嵌入向量
    """
    
    def __init__(self, encoder: Optional[SpecterEncoder] = None, device=None):
        """初始化实体嵌入器
        
        Args:
            encoder: SPECTER编码器实例。如果为None，则创建新的编码器
            device: 计算设备（仅在encoder为None时使用）
        """
        # 如果未提供编码器，创建新的
        if encoder is None:
            self.encoder = SpecterEncoder(device=device)
        else:
            self.encoder = encoder
        
        # 初始化缓存字典
        # 键为实体ID，值为嵌入向量（numpy array）
        self.cache: Dict[str, np.ndarray] = {}
    
    def encode_entity(self, entity: Entity) -> np.ndarray:
        """编码单个实体为768维向量
        
        功能：
        - 使用实体的text字段进行编码
        - 如果实体已在缓存中，直接返回缓存的嵌入
        - 否则使用SPECTER模型编码并存入缓存
        
        Args:
            entity: 实体对象
            
        Returns:
            768维向量 (numpy array)
        """
        # 检查缓存
        if entity.id in self.cache:
            return self.cache[entity.id]
        
        # 使用实体的text字段进行编码
        # 对于实体，我们将text作为输入文本
        text = entity.text
        
        # 使用tokenizer处理文本
        inputs = self.encoder.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 将输入移动到设备
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        # 前向传播（不计算梯度）
        with torch.no_grad():
            outputs = self.encoder.model(**inputs)
        
        # 提取[CLS]位置的输出
        # last_hidden_state形状: (batch_size, seq_len, hidden_size)
        # 取[:, 0, :]得到[CLS]位置的输出，形状为(batch_size, hidden_size)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 返回一维数组 (768,)
        embedding = embedding.squeeze(0)
        
        # 存入缓存
        self.cache[entity.id] = embedding
        
        return embedding
    
    def encode_entities(
        self,
        entities: List[Entity],
        batch_size: int = 64,  # 增加到64以充分利用GPU（从8增加到64）
        show_progress: bool = True
    ) -> np.ndarray:
        """批量编码实体列表
        
        功能：
        - 批量处理实体列表
        - 利用缓存避免重复计算
        - 批量处理未缓存的实体以提高效率
        
        Args:
            entities: 实体列表
            batch_size: 批次大小，默认为8
            show_progress: 是否显示进度条，默认为True
            
        Returns:
            形状为 (n, 768) 的numpy数组，n为实体数量
        """
        if not entities:
            return np.array([]).reshape(0, 768)
        
        # 分离已缓存和未缓存的实体
        cached_embeddings = []
        uncached_entities = []
        entity_to_index = {}  # 记录每个实体在原始列表中的位置
        
        for idx, entity in enumerate(entities):
            if entity.id in self.cache:
                # 从缓存中获取
                cached_embeddings.append((idx, self.cache[entity.id]))
            else:
                # 需要编码
                uncached_entities.append(entity)
                entity_to_index[entity.id] = idx
        
        # 准备结果数组
        n_entities = len(entities)
        all_embeddings = np.zeros((n_entities, 768))
        
        # 填充已缓存的嵌入
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # 批量编码未缓存的实体
        if uncached_entities:
            # 准备文本列表
            texts = [entity.text for entity in uncached_entities]
            
            # 创建进度条
            if show_progress:
                pbar = tqdm(total=len(uncached_entities), desc="编码实体")
            
            # 按批次处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_entities = uncached_entities[i:i + batch_size]
                
                # 使用tokenizer批量处理
                inputs = self.encoder.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # 将输入移动到设备
                inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
                
                # 前向传播（不计算梯度）
                with torch.no_grad():
                    outputs = self.encoder.model(**inputs)
                
                # 提取[CLS]位置的输出
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # 存入缓存并填充结果数组
                for j, entity in enumerate(batch_entities):
                    embedding = batch_embeddings[j]
                    # 存入缓存
                    self.cache[entity.id] = embedding
                    # 填充到结果数组的对应位置
                    original_idx = entity_to_index[entity.id]
                    all_embeddings[original_idx] = embedding
                
                # 更新进度条
                if show_progress:
                    pbar.update(len(batch_texts))
            
            # 关闭进度条
            if show_progress:
                pbar.close()
        
        return all_embeddings

