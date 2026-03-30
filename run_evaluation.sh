#!/bin/bash
# FALCON系统运行和评估脚本

# 设置参数
DATA_DIR="data"
OUTPUT_DIR="outputs"
NUM_QUERIES=500
TOP_K=50

echo "=========================================="
echo "FALCON系统 - 推理和评估"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "查询数量: $NUM_QUERIES"
echo "推荐数量: $TOP_K"
echo "=========================================="
echo ""

# Step 7: 推理
echo "开始推理 (Step 7)..."
python3 run_full_pipeline.py \
    --step 7 \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --num_queries $NUM_QUERIES \
    --top_k $TOP_K \
    --offline

if [ $? -ne 0 ]; then
    echo "推理失败，退出"
    exit 1
fi

echo ""
echo "推理完成！"
echo ""

# Step 8: 评估
echo "开始评估 (Step 8)..."
python3 run_full_pipeline.py \
    --step 8 \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --num_queries $NUM_QUERIES \
    --top_k $TOP_K \
    --offline

if [ $? -ne 0 ]; then
    echo "评估失败，退出"
    exit 1
fi

echo ""
echo "=========================================="
echo "所有步骤完成！"
echo "=========================================="
