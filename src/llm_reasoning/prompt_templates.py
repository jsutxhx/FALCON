"""提示词模板模块"""
from typing import Dict


# 推荐提示词模板
RECOMMENDATION_PROMPT = """You are an academic citation recommendation expert. Your task is to recommend relevant papers for citation based on the query paper and candidate papers.

## Query Paper Information
- Title: {query_title}
- Abstract: {query_abstract}
- Citation Function: {citation_function}

## Candidate Papers
{candidate_list}

## Reasoning Instructions
Please follow these four phases to analyze and recommend papers:

### Phase 1: Query Analysis
Extract and analyze:
- Research task: What is the main research problem or task?
- Methodological characteristics: What methods or approaches are used?
- Citation intent: What is the purpose of citing papers (background, use, compare, or inspire)?

### Phase 2: Candidate Analysis
For each candidate paper, analyze:
- Knowledge entities: What are the key entities (tasks, methods, materials, metrics)?
- Quality characteristics: What is the paper's quality (citations, venue, recency)?
- Function match: How well does the paper match the citation function?

### Phase 3: Matching Evaluation
Evaluate multi-dimensional matching between query and candidates:
- Entity similarity: How similar are the knowledge entities?
- Path similarity: Are there semantic paths connecting the papers?
- Function-specific match: Does the candidate serve the intended citation function?

### Phase 4: Recommendation Generation
Generate recommendations with the following information for each recommended paper:
- Recommended paper ID: The unique identifier of the paper
- Recommendation reason: A clear explanation of why this paper is recommended
- Suggested citation position: Where in the paper this citation should be placed (introduction, methodology, experiment, or discussion)
- Confidence level: Your confidence in this recommendation (high, medium, or low)

## Output Format
Return a JSON array with the following structure:
```json
[
  {{
    "paper_id": "paper_id_string",
    "reason": "Detailed explanation of why this paper is recommended",
    "citation_position": "introduction|methodology|experiment|discussion",
    "confidence": "high|medium|low"
  }}
]
```

Please provide your recommendations in the JSON format above. Focus on papers that best match the citation function and provide clear, actionable reasons for each recommendation.
"""


# 功能描述字典（用于在提示词中提供更详细的功能说明）
FUNCTION_DESCRIPTIONS: Dict[str, str] = {
    "background": (
        "Background citations provide context and establish the research foundation. "
        "Look for papers that introduce the research area, define key concepts, or provide "
        "theoretical background relevant to the query paper."
    ),
    "use": (
        "Use citations indicate that the query paper adopts or builds upon methods, techniques, "
        "or tools from the cited paper. Look for papers with similar methodologies, algorithms, "
        "or technical approaches that can be directly applied or extended."
    ),
    "compare": (
        "Compare citations involve comparing different approaches, methods, or results. "
        "Look for papers that address similar tasks but use different methods, allowing "
        "for meaningful comparison and evaluation."
    ),
    "inspire": (
        "Inspire citations refer to papers from different domains that provide novel ideas, "
        "concepts, or perspectives that can inspire new research directions. Look for papers "
        "with transferable ideas or cross-domain insights."
    )
}


def get_function_description(function: str) -> str:
    """获取引用功能的详细描述
    
    Args:
        function: 引用功能类型（background, use, compare, inspire）
        
    Returns:
        功能描述字符串，如果功能类型未知则返回默认描述
    """
    return FUNCTION_DESCRIPTIONS.get(function.lower(), 
                                     "General citation recommendation based on relevance and quality.")


