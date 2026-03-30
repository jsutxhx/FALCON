"""文本预处理器模块"""
import re
import unicodedata


class TextPreprocessor:
    """文本预处理器"""
    
    def clean_text(self, text: str) -> str:
        """清洗文本
        
        功能：
        - 去除多余空白
        - 统一 Unicode 字符
        - 去除特殊控制字符
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 统一 Unicode 字符（NFKC规范化）
        text = unicodedata.normalize('NFKC', text)
        
        # 去除特殊控制字符（保留换行符、制表符等常用空白字符）
        # 移除控制字符（除了常见的空白字符）
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        
        # 去除多余空白
        # 将多个连续空格替换为单个空格（保留制表符和换行符）
        text = re.sub(r' +', ' ', text)  # 多个空格替换为单个空格
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个换行保留最多两个
        text = re.sub(r' +\n', '\n', text)  # 行尾空格
        text = re.sub(r'\n +', '\n', text)  # 行首空格
        text = re.sub(r' +\t', '\t', text)  # 制表符前的空格
        text = re.sub(r'\t +', '\t', text)  # 制表符后的空格
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def split_sentences(self, text: str) -> list:
        """将文本分割为句子
        
        功能：
        - 使用正则表达式处理句末标点
        - 处理缩写词 (Dr., Mr., etc.)
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if not text or not text.strip():
            return []
        
        # 常见缩写词模式（用于保护，避免被误判为句子结束）
        abbreviations = [
            r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Ph\.D\.',
            r'U\.S\.A\.', r'U\.S\.', r'etc\.', r'e\.g\.', r'i\.e\.',
            r'vs\.', r'Inc\.', r'Ltd\.', r'Corp\.', r'St\.', r'Ave\.',
            r'Jan\.', r'Feb\.', r'Mar\.', r'Apr\.', r'Jun\.', r'Jul\.',
            r'Aug\.', r'Sep\.', r'Oct\.', r'Nov\.', r'Dec\.'
        ]
        
        # 创建缩写词模式
        abbrev_pattern = '|'.join(abbreviations)
        
        # 临时替换缩写词中的句号，避免被误判为句子结束
        protected_text = text
        abbrev_map = {}
        if abbrev_pattern:
            # 从后往前替换，避免位置偏移
            matches = list(re.finditer(abbrev_pattern, protected_text, re.IGNORECASE))
            for idx, match in enumerate(reversed(matches)):
                placeholder = f"__ABBREV_{len(matches) - idx - 1}__"
                abbrev_map[placeholder] = match.group()
                abbrev_text = match.group()
                # 将句号替换为特殊标记
                protected_text = (
                    protected_text[:match.start()] + 
                    abbrev_text.replace('.', '__DOT__') + 
                    protected_text[match.end():]
                )
        
        # 句子分割模式：句号、问号、感叹号后跟空格、换行或文本结束
        # 支持中英文标点（中文标点后可能没有空格）
        sentence_endings = r'([.!?。！？])(?:\s+|$)'
        
        # 使用finditer找到所有句子结束位置
        sentences = []
        last_end = 0
        
        for match in re.finditer(sentence_endings, protected_text):
            # 提取从上次结束到当前标点的文本
            sentence = protected_text[last_end:match.end()].strip()
            if sentence:
                # 恢复缩写词中的句号
                sentence = sentence.replace('__DOT__', '.')
                sentences.append(sentence)
            last_end = match.end()
        
        # 处理最后一部分（如果没有以标点结尾）
        if last_end < len(protected_text):
            sentence = protected_text[last_end:].strip()
            if sentence:
                # 恢复缩写词中的句号
                sentence = sentence.replace('__DOT__', '.')
                sentences.append(sentence)
        
        # 如果结果为空，返回原文本作为单个句子
        if not sentences:
            return [text.strip()] if text.strip() else []
        
        return sentences
    
    def tokenize(self, text: str) -> list:
        """将文本分割为词（token）
        
        功能：
        - 使用空格分词
        - 处理标点符号，将标点与单词分离
        
        Args:
            text: 输入文本
            
        Returns:
            词（token）列表
        """
        if not text or not text.strip():
            return []
        
        # 在标点符号前后添加空格，以便分离标点和单词
        # 匹配常见标点符号（保留字母、数字、空格）
        # 使用正向先行断言和负向后行断言，避免重复添加空格
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        
        # 使用空格分割
        tokens = text.split()
        
        # 过滤空字符串
        tokens = [token for token in tokens if token.strip()]
        
        return tokens

