"""引用功能分类器模块"""
from typing import Optional, Tuple, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.utils.device_utils import get_device


class CitationFunctionClassifier:
    """引用功能分类器
    
    功能：
    - 使用SciBERT模型对引用上下文进行分类
    - 识别引用的功能类型：background, use, compare, inspire
    
    Attributes:
        FUNCTIONS: 引用功能列表，包含4个类别
        tokenizer: SciBERT tokenizer
        model: BERT模型（用于序列分类）
        device: 计算设备
        max_length: 最大序列长度
    """
    
    # 引用功能列表
    FUNCTIONS = ["background", "use", "compare", "inspire"]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_length: int = 256,
        device: Optional[str] = None,
        num_labels: int = 4
    ):
        """初始化引用功能分类器
        
        Args:
            model_path: 微调后的模型检查点路径。如果为None，则使用预训练模型
            model_name: 预训练模型名称，默认为 "allenai/scibert_scivocab_uncased"
            max_length: 最大序列长度，默认为256
            device: 计算设备 ("cuda", "cpu" 或 None自动检测)
            num_labels: 标签数量，默认为4（对应4个引用功能类别）
        """
        # 验证num_labels
        if num_labels != 4:
            raise ValueError(f"num_labels must be 4 for citation function classification, got {num_labels}")
        
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
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path_obj),
                num_labels=num_labels
            )
        else:
            # 使用预训练模型（需要指定num_labels）
            # 注意：预训练模型通常没有分类头，所以这里需要指定num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        
        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
    
    def classify(self, citation_context: str) -> Tuple[str, Dict[str, float]]:
        """分类引用功能
        
        功能：
        - 对引用上下文进行分类，识别其功能类型
        - 返回预测的功能类型和各类别概率
        
        公式：
        P(f|c) = Softmax(W_f · SciBERT(c)_{[CLS]} + b_f)
        其中：
        - c 是引用上下文
        - f 是功能类型（background, use, compare, inspire）
        - W_f 和 b_f 是分类层的权重和偏置
        
        Args:
            citation_context: 引用上下文字符串，例如 "We adopt the method proposed in..."
            
        Returns:
            (predicted_function, prob_dict) 元组：
            - predicted_function: 预测的功能类型字符串（"background", "use", "compare", "inspire"之一）
            - prob_dict: 各类别概率字典，键为功能类型，值为概率（范围[0, 1]），所有概率之和为1.0
        """
        # Tokenize输入文本
        inputs = self.tokenizer(
            citation_context,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # 将输入移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 模型推理（禁用梯度计算）
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 应用softmax获取概率分布
            probs = F.softmax(logits, dim=-1)[0]  # [0] 取第一个（也是唯一的）样本
        
        # 构建概率字典
        prob_dict = {func: probs[i].item() for i, func in enumerate(self.FUNCTIONS)}
        
        # 获取预测的功能类型（概率最大的类别）
        predicted_idx = probs.argmax().item()
        predicted_function = self.FUNCTIONS[predicted_idx]
        
        return predicted_function, prob_dict

