"""SPECTER论文嵌入编码器模块"""
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from src.utils.device_utils import get_device
from src.data_processing.data_structures import Paper


class SpecterEncoder:
    """SPECTER论文嵌入编码器
    
    功能：
    - 使用SPECTER模型生成论文全局嵌入(768维)
    - 支持单篇论文和批量论文编码
    
    Attributes:
        tokenizer: SPECTER tokenizer
        model: SPECTER模型
        device: 计算设备
    """
    
    def __init__(self, device=None, model_name="allenai/specter"):
        """初始化SPECTER编码器
        
        Args:
            device: 计算设备 ("cuda", "cpu" 或 None自动检测)
            model_name: 模型名称或路径，默认为 "allenai/specter"
        """
        # 获取设备
        self.device = get_device(device)
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    def encode_paper(self, title: str, abstract: str) -> np.ndarray:
        """编码单篇论文为768维向量
        
        功能：
        - 拼接title和abstract
        - 使用SPECTER模型编码
        - 提取[CLS]位置的输出
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            768维向量 (numpy array)
        """
        # 拼接title和abstract
        text = title + " " + abstract
        
        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 将输入移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向传播（不计算梯度）
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 提取[CLS]位置的输出 (batch_size=1, 所以取[0])
        # last_hidden_state形状: (batch_size, seq_len, hidden_size)
        # 取[:, 0, :]得到[CLS]位置的输出，形状为(batch_size, hidden_size)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 返回一维数组 (768,)
        return embedding.squeeze(0)
    
    def encode_papers(
        self,
        papers: List[Paper],
        batch_size: int = 64,  # 增加到64以充分利用GPU（从8增加到64）
        show_progress: bool = True
    ) -> np.ndarray:
        """批量编码多篇论文
        
        功能：
        - 批量处理论文列表
        - 使用批量处理提高效率
        - 支持进度条显示
        
        Args:
            papers: 论文列表
            batch_size: 批次大小，默认为8
            show_progress: 是否显示进度条，默认为True
            
        Returns:
            形状为 (n, 768) 的numpy数组，n为论文数量
        """
        if not papers:
            return np.array([]).reshape(0, 768)
        
        # 准备文本列表
        texts = []
        for paper in papers:
            text = paper.title + " " + paper.abstract
            texts.append(text)
        
        # 批量编码
        embeddings = []
        
        # 创建进度条
        if show_progress:
            pbar = tqdm(total=len(papers), desc="编码论文")
        
        # 按批次处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 使用tokenizer批量处理
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 前向传播（不计算梯度）
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取[CLS]位置的输出
            # last_hidden_state形状: (batch_size, seq_len, hidden_size)
            # 取[:, 0, :]得到[CLS]位置的输出，形状为(batch_size, hidden_size)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
            # 更新进度条
            if show_progress:
                pbar.update(len(batch_texts))
        
        # 关闭进度条
        if show_progress:
            pbar.close()
        
        # 合并所有批次的嵌入
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings

