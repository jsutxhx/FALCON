"""实体链接与规范化模块"""
import re
from typing import Set


class EntityLinker:
    """实体链接与规范化器
    
    功能：
    - 将同义实体规范化到相同的规范形式
    - 使用简单规则：小写、去停用词、词干化
    
    Attributes:
        stopwords: 停用词集合
    """
    
    def __init__(self):
        """初始化实体链接器"""
        # 定义常见停用词（英文）
        self.stopwords: Set[str] = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'now'
        }
    
    def normalize(self, entity_text: str) -> str:
        """规范化实体文本
        
        功能：
        - 转换为小写
        - 去除停用词
        - 简单词干化（去除常见后缀）
        - 去除标点符号和多余空格
        
        Args:
            entity_text: 原始实体文本
            
        Returns:
            规范化后的实体文本（canonical form）
        """
        if not entity_text:
            return ""
        
        # 1. 转换为小写
        normalized = entity_text.lower()
        
        # 2. 去除标点符号（保留字母、数字和空格）
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # 3. 分词
        words = normalized.split()
        
        # 4. 去除停用词
        words = [word for word in words if word not in self.stopwords]
        
        # 5. 简单词干化（去除常见后缀）
        stemmed_words = []
        for word in words:
            stemmed = self._simple_stem(word)
            if stemmed:  # 只保留非空词干
                stemmed_words.append(stemmed)
        
        # 6. 合并单词并去除多余空格
        canonical = ' '.join(stemmed_words).strip()
        
        # 7. 如果结果为空，返回原始文本的小写形式（至少保留一些信息）
        if not canonical:
            # 如果所有词都是停用词，至少返回去除标点的小写形式
            normalized = re.sub(r'[^\w\s]', ' ', entity_text.lower())
            canonical = ' '.join(normalized.split()).strip()
        
        return canonical
    
    def _simple_stem(self, word: str) -> str:
        """简单词干化（去除常见后缀）
        
        功能：
        - 去除常见的英文后缀（-s, -ed, -ing, -er, -est, -ly等）
        - 使用简单的规则，不依赖外部库
        
        Args:
            word: 输入单词
            
        Returns:
            词干
        """
        if len(word) <= 3:
            return word
        
        # 按优先级尝试去除后缀
        suffixes = [
            ('ies', 'y'),      # studies -> studi -> study (需要特殊处理)
            ('ied', 'y'),      # studied -> studi -> study
            ('ing', ''),       # learning -> learn
            ('ed', ''),        # learned -> learn
            ('er', ''),        # learner -> learn
            ('est', ''),       # fastest -> fast
            ('ly', ''),        # quickly -> quick
            ('s', ''),         # networks -> network
            ('es', ''),        # classes -> class
        ]
        
        # 特殊处理：ies -> y, ied -> y
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        if word.endswith('ied') and len(word) > 4:
            return word[:-3] + 'y'
        
        # 尝试其他后缀
        for suffix, replacement in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                # 确保去除后缀后至少还有2个字符
                stemmed = word[:-len(suffix)] + replacement
                if len(stemmed) >= 2:
                    return stemmed
        
        return word


