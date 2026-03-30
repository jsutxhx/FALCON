"""重新构建引用关系脚本"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Set
from loguru import logger
from dataclasses import asdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.data_structures import Citation, Paper
from scripts.data_acquisition import SemanticScholarAPI


def fetch_citations_from_api(
    papers: List[Paper],
    api_key: str = None,
    rate_limit: float = 1.0,
    max_papers: int = None
) -> List[Citation]:
    """从Semantic Scholar API获取引用关系
    
    Args:
        papers: 论文列表
        api_key: API密钥（可选）
        rate_limit: 请求间隔（秒）
        max_papers: 最大处理论文数（用于测试）
        
    Returns:
        引用关系列表
    """
    if max_papers:
        papers = papers[:max_papers]
    
    logger.info(f"开始从API获取 {len(papers)} 篇论文的引用关系...")
    
    api = SemanticScholarAPI(api_key=api_key, rate_limit=rate_limit)
    
    # 创建论文ID到Paper对象的映射
    paper_dict = {paper.id: paper for paper in papers}
    all_paper_ids = set(paper_dict.keys())
    
    citations = []
    citation_count = 0
    failed_count = 0
    
    for i, paper in enumerate(papers, 1):
        if i % 50 == 0:
            logger.info(f"已处理 {i}/{len(papers)} 篇论文，找到 {citation_count} 条引用关系，失败 {failed_count} 次")
        
        try:
            # 获取论文的引用列表
            references = api.get_paper_references(paper.id)
            
            if references:
                for ref in references:
                    # 提取被引用论文的ID
                    if isinstance(ref, dict):
                        cited_paper = ref.get("citedPaper", {})
                        if cited_paper:
                            cited_id = cited_paper.get("paperId") or cited_paper.get("id")
                            
                            # 只保留在我们数据集中的引用关系
                            if cited_id and cited_id in all_paper_ids:
                                citation = Citation(
                                    source_paper_id=paper.id,
                                    target_paper_id=cited_id,
                                    context="",  # API可能不提供上下文
                                    position="introduction"  # 默认位置
                                )
                                citations.append(citation)
                                citation_count += 1
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只记录前5个错误
                logger.warning(f"获取论文 {paper.id} 的引用关系失败: {e}")
            continue
    
    logger.info(f"从API获取了 {len(citations)} 条有效引用关系（失败 {failed_count} 次）")
    return citations


def build_citations_from_existing_data(
    papers_data: List[Dict],
    all_paper_ids: Set[str]
) -> List[Citation]:
    """从现有数据中构建引用关系
    
    检查论文数据中是否已有引用信息（references字段）
    
    Args:
        papers_data: 论文数据列表（字典格式）
        all_paper_ids: 所有论文ID集合
        
    Returns:
        引用关系列表
    """
    logger.info(f"开始从现有数据构建引用关系...")
    
    citations = []
    
    for paper_data in papers_data:
        paper_id = paper_data.get("id")
        if not paper_id:
            continue
        
        # 检查是否有references字段
        references = paper_data.get("references", [])
        if not references:
            continue
        
        # 处理引用列表
        for ref in references:
            if isinstance(ref, dict):
                target_id = ref.get("paper_id") or ref.get("paperId") or ref.get("id")
            else:
                target_id = str(ref)
            
            # 只保留在我们数据集中的引用关系
            if target_id and target_id in all_paper_ids:
                citation = Citation(
                    source_paper_id=paper_id,
                    target_paper_id=target_id,
                    context="",  # 原始数据中可能没有上下文
                    position="introduction"  # 默认位置
                )
                citations.append(citation)
    
    logger.info(f"从现有数据构建了 {len(citations)} 条引用关系")
    return citations


def rebuild_citations(
    papers_path: str,
    output_path: str,
    api_key: str = None,
    use_api: bool = True,
    max_papers: int = None
) -> List[Citation]:
    """重新构建引用关系
    
    Args:
        papers_path: 论文JSON文件路径
        output_path: 输出文件路径
        api_key: Semantic Scholar API密钥（可选）
        use_api: 是否使用API获取引用关系
        max_papers: 最大处理论文数（用于测试）
        
    Returns:
        引用关系列表
    """
    logger.info("=" * 60)
    logger.info("重新构建引用关系")
    logger.info("=" * 60)
    
    # 加载论文数据
    logger.info(f"加载论文数据: {papers_path}")
    with open(papers_path, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    if max_papers:
        papers_data = papers_data[:max_papers]
    
    logger.info(f"加载了 {len(papers_data)} 篇论文")
    
    # 创建所有论文ID集合（用于验证引用有效性）
    all_paper_ids = {p.get("id") for p in papers_data if p.get("id")}
    
    citations = []
    
    # 首先尝试从现有数据构建引用关系
    logger.info("步骤1: 从现有数据中提取引用关系...")
    citations_from_data = build_citations_from_existing_data(papers_data, all_paper_ids)
    citations.extend(citations_from_data)
    logger.info(f"从现有数据中提取了 {len(citations_from_data)} 条引用关系")
    
    # 如果使用API且提供了API密钥，尝试从API获取更多引用关系
    if use_api and api_key:
        logger.info("步骤2: 从API获取引用关系...")
        try:
            # 创建Paper对象用于API调用（排除references字段，因为Paper数据类不支持）
            papers = []
            for p_data in papers_data:
                # 复制数据并移除references字段
                paper_data = {k: v for k, v in p_data.items() if k != "references"}
                papers.append(Paper(**paper_data))
            citations_from_api = fetch_citations_from_api(papers, api_key=api_key, max_papers=max_papers)
            
            # 合并引用关系（去重）
            existing_citations = {(c.source_paper_id, c.target_paper_id) for c in citations}
            new_citations = [
                c for c in citations_from_api 
                if (c.source_paper_id, c.target_paper_id) not in existing_citations
            ]
            citations.extend(new_citations)
            logger.info(f"从API获取了 {len(new_citations)} 条新的引用关系")
        except Exception as e:
            logger.warning(f"从API获取引用关系失败: {e}")
            logger.info("继续使用现有数据中的引用关系")
    
    # 保存引用关系
    logger.info(f"保存 {len(citations)} 条引用关系到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    citations_dict = [asdict(c) for c in citations]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(citations_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"引用关系已保存到: {output_path}")
    
    return citations


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="重新构建引用关系")
    parser.add_argument(
        "--papers",
        type=str,
        required=True,
        help="论文JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Semantic Scholar API密钥（可选）"
    )
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="使用API获取引用关系"
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=1.0,
        help="API请求间隔（秒）"
    )
    
    args = parser.parse_args()
    
    # 从环境变量获取API密钥（如果未提供）
    if not args.api_key:
        import os
        args.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    citations = rebuild_citations(
        papers_path=args.papers,
        output_path=args.output,
        api_key=args.api_key,
        use_api=args.use_api
    )
    
    logger.info(f"完成！共构建了 {len(citations)} 条引用关系")


if __name__ == "__main__":
    main()

