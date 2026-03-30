"""FALCON系统演示脚本 - 创建测试数据并运行推理"""
import sys
import json
from pathlib import Path
from src.pipeline.falcon import FALCON
from src.knowledge_graph.graph_storage import KnowledgeGraph
from src.data_processing.data_structures import Paper
from loguru import logger

# 创建Mock的CitationFunctionClassifier来避免网络问题
class MockFunctionClassifier:
    FUNCTIONS = ["background", "use", "compare", "inspire"]
    def classify(self, citation_context: str):
        return "background", {f: 0.25 for f in self.FUNCTIONS}

def create_test_data():
    """创建测试数据"""
    logger.info("创建测试数据...")
    
    # 创建知识图谱
    graph = KnowledgeGraph()
    
    # 创建查询论文
    query_paper = Paper(
        id="query1",
        title="Query Paper: Machine Learning in Computer Vision",
        abstract="This paper discusses machine learning methods for computer vision tasks including image classification and object detection.",
        authors=["Author1", "Author2"],
        year=2020,
        venue="Conference",
        citation_count=50
    )
    graph.add_paper_node(query_paper)
    
    # 创建候选论文
    candidates = [
        Paper(
            id="candidate1",
            title="Deep Learning for Image Classification",
            abstract="We propose a deep learning approach for image classification using convolutional neural networks.",
            authors=["Author3"],
            year=2021,
            venue="Conference",
            citation_count=100
        ),
        Paper(
            id="candidate2",
            title="Neural Networks in Vision",
            abstract="Neural networks have shown great success in vision tasks such as object recognition.",
            authors=["Author4"],
            year=2022,
            venue="Journal",
            citation_count=80
        ),
        Paper(
            id="candidate3",
            title="Transfer Learning for Visual Recognition",
            abstract="We explore transfer learning techniques for improving visual recognition performance.",
            authors=["Author5"],
            year=2021,
            venue="Conference",
            citation_count=60
        )
    ]
    
    for candidate in candidates:
        graph.add_paper_node(candidate)
    
    # 创建论文数据库
    paper_db = {
        "query1": query_paper,
        **{c.id: c for c in candidates}
    }
    
    logger.info(f"创建了 {len(paper_db)} 篇论文")
    
    return graph, paper_db, query_paper

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("FALCON系统演示 - 推理模式")
    logger.info("=" * 60)
    
    # 创建测试数据
    graph, paper_db, query_paper = create_test_data()
    
    # 初始化FALCON系统
    logger.info("\n初始化FALCON系统...")
    falcon = FALCON(
        graph=graph,
        paper_db=paper_db,
        function_classifier=MockFunctionClassifier()
    )
    
    # 运行推理
    logger.info("\n运行推理...")
    citation_context = "This paper uses deep learning methods for image classification tasks."
    
    recommendations = falcon.recommend(
        query_paper=query_paper,
        citation_context=citation_context,
        top_k=5
    )
    
    # 显示结果
    logger.info("\n" + "=" * 60)
    logger.info(f"生成 {len(recommendations)} 个推荐:")
    logger.info("=" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n推荐 {i}:")
        logger.info(f"  论文ID: {rec.paper_id}")
        logger.info(f"  得分: {rec.score:.4f}")
        logger.info(f"  理由: {rec.reason}")
        logger.info(f"  引用位置: {rec.citation_position}")
        logger.info(f"  置信度: {rec.confidence}")
    
    # 保存结果
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "recommendations.json"
    
    recommendations_dict = [rec.to_dict() for rec in recommendations]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n推荐结果已保存到: {output_file}")
    logger.info("\n" + "=" * 60)
    logger.info("演示完成！")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()


