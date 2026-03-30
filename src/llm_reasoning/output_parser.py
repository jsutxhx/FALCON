"""输出解析器模块"""
import re
import json
from typing import List
from src.data_processing.data_structures import Recommendation


class OutputParser:
    """LLM输出解析器
    
    功能：
    - 解析LLM生成的文本响应
    - 提取JSON格式的推荐信息
    - 转换为Recommendation对象列表
    - 处理各种解析失败情况
    
    Attributes:
        None
    """
    
    def __init__(self):
        """初始化输出解析器"""
        pass
    
    def parse(self, response: str) -> List[Recommendation]:
        """解析LLM响应为Recommendation对象列表
        
        功能：
        - 从响应文本中提取JSON数组
        - 解析JSON数据
        - 创建Recommendation对象
        - 处理解析失败的情况
        
        算法：
        1. 使用正则表达式查找JSON数组（匹配 [...] 模式）
        2. 尝试解析JSON
        3. 验证每个推荐项的字段
        4. 创建Recommendation对象（如果缺少score，使用默认值0.0）
        5. 过滤掉无效的推荐项
        
        Args:
            response: LLM生成的文本响应
            
        Returns:
            Recommendation对象列表
            如果解析失败或没有有效推荐，返回空列表
        """
        if not response or not isinstance(response, str):
            return []
        
        # 1. 提取JSON部分
        # 使用正则表达式查找JSON数组，支持多行和嵌套
        json_pattern = r'\[[\s\S]*?\]'
        json_match = re.search(json_pattern, response)
        
        if not json_match:
            # 如果没有找到数组，尝试查找单个对象
            object_pattern = r'\{[\s\S]*?\}'
            object_match = re.search(object_pattern, response)
            if object_match:
                # 将单个对象包装成数组
                json_str = f"[{object_match.group()}]"
            else:
                return []
        else:
            json_str = json_match.group()
        
        # 2. 解析JSON
        try:
            data = json.loads(json_str)
            
            # 确保data是列表
            if not isinstance(data, list):
                # 如果是单个对象，转换为列表
                data = [data] if isinstance(data, dict) else []
            
        except json.JSONDecodeError:
            # JSON解析失败，返回空列表
            return []
        
        # 3. 创建Recommendation对象
        recommendations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            try:
                # 提取必需字段
                paper_id = item.get("paper_id", "")
                reason = item.get("reason", "")
                citation_position = item.get("citation_position", "introduction")
                confidence = item.get("confidence", "medium")
                
                # score字段是可选的（LLM可能不生成）
                # 如果没有提供，使用默认值0.0
                score = item.get("score", 0.0)
                
                # 验证必需字段
                if not paper_id:
                    continue
                
                # 创建Recommendation对象
                # 注意：Recommendation的__post_init__会验证citation_position和confidence
                recommendation = Recommendation(
                    paper_id=paper_id,
                    score=float(score),
                    reason=reason,
                    citation_position=citation_position,
                    confidence=confidence
                )
                
                recommendations.append(recommendation)
                
            except (ValueError, KeyError, TypeError) as e:
                # 跳过无效的推荐项
                continue
        
        return recommendations


