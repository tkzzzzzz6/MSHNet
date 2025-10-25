#!/bin/bash
# Google Colab A100 快速启动脚本

echo "========================================="
echo "MSHNet - Colab A100 快速启动"
echo "========================================="

# 检查GPU
echo "步骤1: 检查GPU信息..."
nvidia-smi

# 安装依赖
echo "步骤2: 安装依赖..."
pip install -q scikit-image tqdm

# 检查数据集
echo "步骤3: 检查数据集..."
if [ -d "./datasets/IRSTD-1k" ]; then
    echo "✓ 数据集目录存在"
    ls -lh ./datasets/IRSTD-1k/
else
    echo "✗ 数据集目录不存在，请先准备数据集"
    exit 1
fi

# 开始训练
echo "步骤4: 开始训练..."
python main.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 16 \
    --epochs 400 \
    --lr 0.05 \
    --base-size 256 \
    --crop-size 256 \
    --warm-epoch 5 \
    --mode train

echo "========================================="
echo "训练完成！"
echo "========================================="

