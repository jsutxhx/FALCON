"""从Semantic Scholar API获取OpenCorpus数据"""
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


class SemanticScholarAPI:
    """Semantic Scholar API客户端"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: str = None, rate_limit: float = 1.0):
        """初始化API客户端
        
        Args:
            api_key: API密钥（可选，有密钥可以提高速率限制）
            rate_limit: 请求间隔（秒），默认1.0秒（有API密钥时为1秒/请求）
        """
        self.api_key = api_key
        self.rate_limit = rate_limit if api_key else max(rate_limit, 0.5)  # 无密钥时至少0.5秒
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
    
    def search_papers(
        self,
        query: str,
        fields: List[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """搜索论文
        
        Args:
            query: 搜索查询
            fields: 返回的字段列表
            limit: 返回结果数量
            offset: 偏移量
            
        Returns:
            API响应字典
        """
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors", "year",
                "venue", "citationCount"
            ]
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset
        }
        
        time.sleep(self.rate_limit)
        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_paper(self, paper_id: str, fields: List[str] = None) -> Dict[str, Any]:
        """获取单篇论文详情
        
        Args:
            paper_id: 论文ID
            fields: 返回的字段列表
            
        Returns:
            论文数据字典
        """
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors", "year",
                "venue", "citationCount"
            ]
        
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(fields)}
        
        time.sleep(self.rate_limit)
        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_paper_references(self, paper_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取论文的引用列表
        
        Args:
            paper_id: 论文ID
            limit: 最大返回数量
            
        Returns:
            引用论文列表
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            "fields": "paperId,title,year",
            "limit": limit
        }
        
        time.sleep(self.rate_limit)
        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [ref.get("citedPaper", {}) for ref in data.get("data", []) if ref.get("citedPaper")]


def fetch_opencorpus_data(
    target_count: int = 70000,
    output_path: str = "data/raw/papers/opencorpus_papers.json",
    api_key: str = None,
    query: str = "computer science"
) -> None:
    """从Semantic Scholar获取OpenCorpus数据
    
    Args:
        target_count: 目标论文数量
        output_path: 输出文件路径
        api_key: API密钥（可选）
        query: 搜索查询
    """
    logger.info(f"开始获取OpenCorpus数据，目标数量: {target_count}")
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化API客户端（有API密钥时使用1秒间隔，符合API限制）
    api = SemanticScholarAPI(api_key=api_key, rate_limit=1.0)
    
    papers = []
    offset = 0
    batch_size = 100
    max_offset = 900 - batch_size  # API限制：offset + limit < 1000
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while len(papers) < target_count and offset <= max_offset:
        try:
            logger.info(f"获取论文 {len(papers)}/{target_count} (offset={offset})")
            logger.debug(f"发送API请求...")
            
            # 搜索论文
            search_results = api.search_papers(
                query=query,
                limit=batch_size,
                offset=offset
            )
            
            logger.debug(f"收到API响应，处理数据...")
            
            paper_list = search_results.get("data", [])
            if not paper_list:
                logger.warning("没有更多论文，停止获取")
                break
            
            # 处理每篇论文
            for paper_data in paper_list:
                if len(papers) >= target_count:
                    break
                
                # 提取论文信息
                paper = {
                    "id": paper_data.get("paperId", ""),
                    "title": paper_data.get("title", ""),
                    "abstract": paper_data.get("abstract", ""),
                    "authors": [
                        author.get("name", "") for author in paper_data.get("authors", [])
                    ],
                    "year": paper_data.get("year"),
                    "venue": paper_data.get("venue", ""),
                    "citation_count": paper_data.get("citationCount", 0),
                    "references": []  # 稍后填充
                }
                
                # 只保留有摘要的论文
                if paper["abstract"] and paper["id"]:
                    papers.append(paper)
                    logger.debug(f"添加论文: {paper['id']} - {paper['title'][:50]}")
            
            offset += batch_size
            consecutive_errors = 0  # 重置错误计数
            
            # 保存中间结果（每1000篇保存一次）
            if len(papers) % 1000 == 0:
                logger.info(f"保存中间结果: {len(papers)} 篇论文")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, ensure_ascii=False, indent=2)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                consecutive_errors += 1
                logger.error(f"API访问被拒绝（403 Forbidden）")
                logger.error(f"可能的原因：")
                logger.error(f"  1. 需要API密钥 - 使用 --api_key 参数提供密钥")
                logger.error(f"  2. 请求频率过高 - 请稍后再试")
                logger.error(f"  3. IP被封禁 - 请更换网络或等待")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"连续{max_consecutive_errors}次403错误，停止获取")
                    logger.error(f"建议：")
                    logger.error(f"  1. 获取API密钥: https://www.semanticscholar.org/product/api")
                    logger.error(f"  2. 使用已有的数据文件（跳过Step 1）")
                    logger.error(f"  3. 使用其他数据源（OpenCorpus数据集）")
                    
                    # 如果已经有部分数据，保存它们
                    if papers:
                        logger.info(f"保存已获取的 {len(papers)} 篇论文")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(papers, f, ensure_ascii=False, indent=2)
                    
                    raise RuntimeError("API访问被拒绝，无法继续获取数据")
                
                logger.warning(f"等待30秒后重试（连续错误: {consecutive_errors}/{max_consecutive_errors}）")
                time.sleep(30)  # 403错误时等待更长时间
            elif e.response.status_code == 429:
                # 速率限制错误，使用指数退避
                consecutive_errors += 1
                wait_time = min(30 * (2 ** (consecutive_errors - 1)), 300)  # 最多等待300秒
                logger.warning(f"速率限制（429），等待{wait_time}秒后重试（连续错误: {consecutive_errors}/{max_consecutive_errors}）")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"连续{max_consecutive_errors}次速率限制错误")
                    logger.warning("建议：稍后再试，或减少batch_size")
                    
                    # 保存已获取的数据
                    if papers:
                        logger.info(f"保存已获取的 {len(papers)} 篇论文")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(papers, f, ensure_ascii=False, indent=2)
                    
                    raise RuntimeError(f"遇到速率限制，已保存{len(papers)}篇论文。建议稍后再继续。")
                
                time.sleep(wait_time)
            elif e.response.status_code == 400:
                # 400错误，可能是offset限制（API限制：offset + limit < 1000）
                error_text = ""
                try:
                    error_data = e.response.json()
                    error_text = error_data.get("error", "")
                except:
                    error_text = str(e)
                
                if "offset + limit must be < 1000" in error_text or offset >= 900:
                    logger.warning(f"API offset限制：offset + limit必须 < 1000（当前offset={offset}, limit={batch_size}）")
                    logger.info(f"已获取 {len(papers)} 篇论文，保存数据并停止")
                    logger.warning("提示：Semantic Scholar API建议使用 /paper/search/bulk 或 Datasets API 获取更多数据")
                    
                    # 保存已获取的数据
                    if papers:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(papers, f, ensure_ascii=False, indent=2)
                        logger.info(f"数据已保存到: {output_path}")
                    
                    break  # 退出循环
                else:
                    consecutive_errors += 1
                    logger.error(f"HTTP错误 400: {e}")
                    if consecutive_errors >= max_consecutive_errors:
                        # 保存已获取的数据
                        if papers:
                            logger.info(f"保存已获取的 {len(papers)} 篇论文")
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(papers, f, ensure_ascii=False, indent=2)
                        raise
                    time.sleep(5)
            else:
                consecutive_errors += 1
                logger.error(f"HTTP错误 {e.response.status_code}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    # 保存已获取的数据
                    if papers:
                        logger.info(f"保存已获取的 {len(papers)} 篇论文")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(papers, f, ensure_ascii=False, indent=2)
                    raise
                time.sleep(5)
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"获取数据时出错: {e}")
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"连续{max_consecutive_errors}次错误，停止获取")
                raise
            time.sleep(5)  # 出错后等待
            continue
    
    # 获取引用列表（可选，因为API限制可能较慢）
    logger.info("开始获取引用列表...")
    for i, paper in enumerate(papers):
        if i % 100 == 0:
            logger.info(f"处理引用 {i}/{len(papers)}")
        
        try:
            references = api.get_paper_references(paper["id"], limit=100)
            # 确保references是列表且不为None
            if references and isinstance(references, list):
                paper["references"] = [
                    {
                        "paper_id": ref.get("paperId", ""),
                        "title": ref.get("title", ""),
                        "year": ref.get("year")
                    }
                    for ref in references if ref and isinstance(ref, dict) and ref.get("paperId")
                ]
            else:
                paper["references"] = []
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # 速率限制，等待更长时间
                logger.warning(f"获取论文 {paper['id']} 的引用时遇到速率限制，跳过")
            else:
                logger.warning(f"获取论文 {paper['id']} 的引用失败: {e}")
            paper["references"] = []
        except Exception as e:
            logger.warning(f"获取论文 {paper['id']} 的引用失败: {e}")
            paper["references"] = []
    
    # 保存最终结果
    logger.info(f"保存最终结果: {len(papers)} 篇论文")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据获取完成，保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从Semantic Scholar获取OpenCorpus数据")
    parser.add_argument(
        "--target_count",
        type=int,
        default=70000,
        help="目标论文数量"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/papers/opencorpus_papers.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Semantic Scholar API密钥（可选）"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="computer science",
        help="搜索查询"
    )
    
    args = parser.parse_args()
    
    fetch_opencorpus_data(
        target_count=args.target_count,
        output_path=args.output,
        api_key=args.api_key,
        query=args.query
    )

