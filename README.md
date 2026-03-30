# FALCON: Fine-grained Academic Literature Citation Organization Network

FALCON is a knowledge graph-based academic citation recommendation system that integrates multi-hop graph retrieval, citation function-aware reranking, and large language model reasoning to provide accurate, diverse, and explainable paper recommendations.

## Overview

FALCON addresses the challenge of finding relevant academic papers for citation by leveraging:
- **Multi-hop Graph Retrieval**: Entity similarity and path similarity based on knowledge graphs
- **Citation Function-Aware Reranking**: Intelligent reranking based on citation context classification (background, use, compare, inspire)
- **LLM Reasoning**: Large language model-based recommendation generation with explanations

## System Architecture

The FALCON system consists of the following key components:

1. **Data Processing**: Data loading, preprocessing, and format conversion
2. **Knowledge Graph**: Graph construction, storage, and querying
3. **Embeddings**: Paper and entity embedding vector generation using SPECTER and SciBERT
4. **Retrieval**: Multi-hop graph retrieval with entity similarity and path similarity
5. **Reranking**: Citation function classification, dynamic weight fusion, cognitive and quality scoring
6. **LLM Reasoning**: Prompt construction, chain-of-thought reasoning, output parsing, and fact verification
7. **Evaluation**: Accuracy, diversity, explainability, and functionality metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FALCON-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

The system provides a complete pipeline script that handles all steps from data acquisition to evaluation:

```bash
# Run inference (Step 7)
python3 run_full_pipeline.py \
    --step 7 \
    --data_dir data \
    --output_dir outputs \
    --num_queries 500 \
    --top_k 50 \
    --offline

# Run evaluation (Step 8)
python3 run_full_pipeline.py \
    --step 8 \
    --data_dir data \
    --output_dir outputs \
    --num_queries 500 \
    --top_k 50 \
    --offline
```

### Complete Pipeline

The full pipeline consists of 8 steps:

1. **Data Acquisition**: Fetch papers from OpenAlex API
2. **Data Preprocessing**: Clean and format the data
3. **Entity & Relation Extraction**: Extract entities and relations from papers
4. **Knowledge Graph Construction**: Build the knowledge graph
5. **Embedding Computation**: Compute paper and entity embeddings
6. **Model Training**: Train citation function classifier and other models
7. **Inference**: Generate recommendations for queries
8. **Evaluation**: Evaluate the system performance

Run all steps:
```bash
python3 run_full_pipeline.py --steps all --data_dir data --output_dir outputs
```

Run specific steps:
```bash
python3 run_full_pipeline.py --steps 7,8 --data_dir data --output_dir outputs
```

### Command Line Arguments

#### Common Arguments
- `--data_dir`: Data directory path (default: `data`)
- `--output_dir`: Output directory path (default: `outputs`)
- `--offline`: Offline mode using mock components (for testing)

#### Step 7 (Inference) Arguments
- `--num_queries`: Number of queries to process (default: 500)
- `--top_k`: Number of recommendations to return (default: 50)

#### Step 8 (Evaluation) Arguments
- `--num_queries`: Number of queries to evaluate (default: 500)
- `--top_k`: Number of recommendations per query (default: 50)

## Evaluation Metrics

FALCON evaluates recommendations across four dimensions:

### Accuracy Metrics
- **P@K**: Precision at K
- **R@K**: Recall at K
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

### Diversity Metrics
- **ILD**: Intra-List Distance
- **Topic Coverage**: Coverage of different research topics
- **Temporal Diversity**: Temporal distribution of recommended papers

### Explainability Metrics
- **Path Coverage**: Percentage of recommendations with explainable paths in the knowledge graph
- **Evidence Verifiability**: Percentage of recommended papers that exist in the paper database

### Function Adaptability Metrics
- **FMA_background**: Function match accuracy for background citations
- **FMA_use**: Function match accuracy for use citations
- **FMA_compare**: Function match accuracy for compare citations
- **FMA_inspire**: Function match accuracy for inspire citations
- **FMA_overall**: Overall function match accuracy

## Experimental Results

### Performance on Test Set (500 queries, top_k=50)

#### Accuracy Metrics
- **P@5**: 1.0000
- **R@5**: 0.1128
- **P@10**: 1.0000
- **R@10**: 0.2256
- **P@20**: 0.9901
- **R@20**: 0.4351
- **MAP**: 0.8317
- **MRR**: 1.0000

#### Diversity Metrics
- **ILD**: 0.8884
- **Topic Coverage**: 1.0000
- **Temporal Diversity**: 3.9119

#### Explainability Metrics
- **Path Coverage**: 0.5107
- **Evidence Verifiability**: 0.5570

#### Function Adaptability Metrics
- **FMA_background**: 1.0000
- **FMA_use**: 1.0000
- **FMA_compare**: 1.0000
- **FMA_inspire**: 1.0000
- **FMA_overall**: 1.0000

### Ablation Study Results

The ablation study evaluates the contribution of each component to the overall system QoE (Quality of Experience):

| Experiment | QoE | Description |
|------------|-----|-------------|
| Full System | 0.7020 | Complete system with all components |
| w/o KG | 0.6050 | Without knowledge graph |
| w/o Reranking | 0.6681 | Without reranking module |
| w/o LLM | 0.6073 | Without LLM reasoning |
| w/o Diversity | 0.6433 | Without diversity optimization |
| w/o Explainability | 0.6165 | Without explainability features |
| Baseline | 0.4504 | Baseline retrieval-only system |

The ablation study demonstrates that each component contributes significantly to the overall system performance, with the knowledge graph and LLM reasoning being the most critical components.

## Project Structure

```
FALCON-main/
├── main.py                      # Main entry point
├── run_full_pipeline.py         # Complete pipeline script
├── run_evaluation.sh            # Evaluation automation script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── config/                      # Configuration files
│   ├── base_config.yaml
│   ├── graph_config.yaml
│   └── model_config.yaml
├── data/                        # Data directory
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── knowledge_graph/         # Knowledge graph files
├── src/                         # Source code
│   ├── data_processing/         # Data processing modules
│   ├── knowledge_graph/         # Knowledge graph modules
│   ├── embeddings/              # Embedding modules
│   ├── retrieval/               # Retrieval modules
│   ├── reranking/               # Reranking modules
│   ├── llm_reasoning/           # LLM reasoning modules
│   ├── evaluation/              # Evaluation modules
│   ├── pipeline/                 # Pipeline integration
│   └── utils/                   # Utility functions
├── outputs/                     # Output results
│   ├── recommendations/         # Recommendation results
│   ├── evaluations/             # Evaluation results
│   └── visualizations/         # Visualization charts
├── scripts/                      # Utility scripts
└── logs/                        # Log files
```

## Key Features

### 1. Multi-hop Graph Retrieval
- Entity similarity calculation based on knowledge graph embeddings
- Path similarity calculation using graph traversal
- Hybrid retrieval combining entity and path similarities

### 2. Citation Function-Aware Reranking
- Automatic citation function classification (background, use, compare, inspire)
- Dynamic weight fusion based on citation context
- Cognitive scoring considering user intent
- Quality scoring based on paper characteristics

### 3. LLM-Based Reasoning
- Chain-of-thought reasoning for recommendation generation
- Natural language explanations for recommendations
- Fact verification to ensure recommended papers exist
- Citation position suggestions

### 4. Comprehensive Evaluation
- Multi-dimensional evaluation (accuracy, diversity, explainability, functionality)
- Detailed debugging information for analysis
- Ablation study support

## Data Format

### Input Data
The system expects data in the following format:

```json
{
  "paper_id": "string",
  "title": "string",
  "abstract": "string",
  "authors": ["string"],
  "year": int,
  "citations": ["paper_id"]
}
```

### Output Format
Recommendations are saved in JSON format:

```json
{
  "query_id": "string",
  "recommendations": [
    {
      "paper_id": "string",
      "score": float,
      "reason": "string",
      "citation_function": "string"
    }
  ]
}
```

## Configuration

Configuration files are located in the `config/` directory:

- `base_config.yaml`: Basic project configuration
- `graph_config.yaml`: Knowledge graph settings
- `model_config.yaml`: Model hyperparameters

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--num_queries` or use `--max_papers` to limit data size
2. **Slow Inference**: Enable GPU acceleration or reduce `--top_k`
3. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Offline Mode

For testing without network access, use the `--offline` flag:
```bash
python3 run_full_pipeline.py --step 7 --offline
```

This uses mock components (MockLLMClient) that simulate LLM responses without making actual API calls.

## Citation

If you use FALCON in your research, please cite:

```bibtex
@article{falcon2024,
  title={FALCON: Fine-grained Academic Literature Citation Organization Network},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add license information here]

## Contact

[Add contact information here]

## Acknowledgments

This project uses the following open-source libraries and models:
- SPECTER for paper embeddings
- SciBERT for entity extraction
- NetworkX for graph operations
- Transformers for language models
