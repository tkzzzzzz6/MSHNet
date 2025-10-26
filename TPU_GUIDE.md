# Google Colab TPU 训练指南

本指南详细介绍如何使用Google Colab的TPU训练MSHNet模型。

## 📋 目录
1. [TPU简介](#tpu简介)
2. [代码修改要点](#代码修改要点)
3. [使用步骤](#使用步骤)
4. [性能对比](#性能对比)
5. [常见问题](#常见问题)

---

## 🔍 TPU简介

### 什么是TPU？
- **TPU (Tensor Processing Unit)**：Google开发的AI加速芯片
- **专为深度学习优化**：特别适合大batch size训练
- **Colab提供**：TPU v2-8（8个核心）或 TPU v3-8

### TPU vs GPU对比

| 特性 | TPU v2-8 | A100 GPU | T4 GPU |
|-----|----------|----------|--------|
| 计算单元 | 8核心 | 1个GPU | 1个GPU |
| 内存 | 64GB HBM | 40GB | 16GB |
| 峰值算力 | 420 TFLOPS | 312 TFLOPS | 65 TFLOPS |
| 适合场景 | 大batch训练 | 通用训练 | 推理/小模型 |
| Colab可用性 | 免费/Pro | 仅Pro | 免费 |

---

## 🔧 代码修改要点

### 1. 核心修改内容

#### 导入TPU库
```python
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
```

#### 设备设置
```python
# GPU方式（原来）
device = torch.device('cuda')

# TPU方式（修改后）
device = xm.xla_device()  # 自动选择TPU设备
```

#### 数据加载
```python
# TPU需要使用ParallelLoader
train_loader = pl.ParallelLoader(
    self.train_loader, 
    [self.device]
).per_device_loader(self.device)
```

#### 优化器步骤
```python
# GPU方式
self.optimizer.step()

# TPU方式
xm.optimizer_step(self.optimizer)  # 使用XLA优化器
```

#### 模型保存
```python
# GPU方式
torch.save(model.state_dict(), 'model.pkl')

# TPU方式
xm.save(model.state_dict(), 'model.pkl')  # 使用XLA保存
```

### 2. 完整的修改对照

| 功能 | GPU代码 | TPU代码 |
|-----|---------|---------|
| 设备 | `torch.device('cuda')` | `xm.xla_device()` |
| 数据加载 | `DataLoader(...)` | `pl.ParallelLoader(...)` |
| 优化器 | `optimizer.step()` | `xm.optimizer_step(optimizer)` |
| 保存 | `torch.save(...)` | `xm.save(...)` |
| 多核 | `DataParallel` | `xmp.spawn(...)` |

---

## 🚀 使用步骤

### 步骤1：选择TPU运行时

1. 打开Colab笔记本
2. 点击：**运行时 → 更改运行时类型**
3. 硬件加速器：选择 **TPU**
4. 点击**保存**

### 步骤2：安装TPU支持库

```python
# 安装PyTorch XLA
!pip install cloud-tpu-client torch-xla torchvision

# 验证安装
import torch_xla
import torch_xla.core.xla_model as xm
print(f'TPU设备: {xm.xla_device()}')
print(f'TPU核心数: {xm.xrt_world_size()}')
```

### 步骤3：上传TPU版本代码

将 `main_tpu.py` 上传到项目目录，或者将原 `main.py` 替换为TPU版本。

### 步骤4：准备数据集

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p datasets
!ln -s /content/drive/MyDrive/datasets/IRSTD-1k ./datasets/IRSTD-1k
```

### 步骤5：开始训练

```bash
# 单核TPU训练
python main_tpu.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 32 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --use-tpu

# 多核TPU训练（8核）
python main_tpu.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 32 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --use-tpu \
    --num-cores 8
```

---

## 📊 性能对比

### 训练速度对比（IRSTD-1k数据集）

| 硬件 | Batch Size | 单Epoch时间 | 400 Epochs总时间 | 内存使用 |
|-----|-----------|------------|----------------|---------|
| **TPU v2-8** | 32 | ~30秒 | **~3-4小时** | 45GB |
| A100 GPU | 16 | ~60秒 | ~7小时 | 20GB |
| T4 GPU | 4 | ~180秒 | ~20小时 | 12GB |
| CPU | 1 | ~600秒 | ~67小时 | 8GB |

### TPU优势

✅ **速度优势**：比T4快6倍，比A100快2倍  
✅ **内存优势**：64GB HBM，支持更大batch size  
✅ **成本优势**：Colab免费提供（有限时）  
✅ **批处理优势**：大batch size训练效果更好

### TPU劣势

❌ **首次编译慢**：第一个epoch需要编译，可能需要5-10分钟  
❌ **调试困难**：错误信息不如GPU清晰  
❌ **生态限制**：部分PyTorch操作不支持  
❌ **可用性限制**：Colab TPU可能需要排队

---

## ⚠️ 注意事项

### 1. Batch Size选择

```python
# TPU推荐的batch size
- TPU v2-8: 32-64
- TPU v3-8: 64-128

# 原因：TPU对大batch size有优化
```

### 2. 数据加载优化

```python
# 使用多进程加载数据
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # TPU推荐4-8个workers
    drop_last=True
)
```

### 3. 首次编译时间

- **第一个epoch**：可能需要5-10分钟（编译）
- **后续epochs**：正常速度
- **建议**：使用小epochs先测试

### 4. 模型保存

```python
# TPU保存模型需要使用xm.save
import torch_xla.core.xla_model as xm

# 保存到本地
xm.save(model.state_dict(), './weight/model.pkl')

# 保存到Drive（推荐）
xm.save(model.state_dict(), '/content/drive/MyDrive/weight/model.pkl')
```

### 5. 调试建议

```python
# 启用XLA调试信息
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'

# 查看编译信息
import torch_xla.debug.metrics as met
print(met.metrics_report())
```

---

## 🔧 常见问题

### Q1: TPU初始化失败？

**错误**：`RuntimeError: Couldn't find TPU`

**解决**：
1. 确认运行时类型选择了TPU
2. 重启运行时：运行时 → 重启运行时
3. 检查XLA安装：`pip install --upgrade torch-xla`

### Q2: 首个epoch特别慢？

**原因**：TPU需要编译计算图

**解决**：正常现象，后续epochs会快很多

### Q3: 内存溢出？

**错误**：`OutOfMemoryError`

**解决**：
```python
# 减小batch size
--batch-size 16  # 从32减到16

# 或减小图像尺寸
--base-size 224  # 从256减到224
```

### Q4: 数据加载慢？

**解决**：
```python
# 增加workers数量
DataLoader(..., num_workers=8)

# 使用内存缓存
trainset = IRSTD_Dataset(..., cache=True)
```

### Q5: 模型保存失败？

**错误**：`XLA tensor on different device`

**解决**：
```python
# 使用xm.save而不是torch.save
import torch_xla.core.xla_model as xm
xm.save(model.state_dict(), 'model.pkl')
```

### Q6: TPU vs GPU，选哪个？

| 场景 | 推荐 | 原因 |
|-----|------|------|
| 大模型训练 | **TPU** | 内存大，速度快 |
| 小模型训练 | GPU | 启动快，调试方便 |
| 实验调试 | GPU | 错误信息清晰 |
| 生产训练 | TPU | 速度快，成本低 |
| 首次尝试 | GPU | 兼容性好 |

---

## 📝 完整训练脚本

### 单核TPU训练

```python
# colab_tpu_single.py
import torch
import torch_xla.core.xla_model as xm

# 1. 检查TPU
device = xm.xla_device()
print(f'使用设备: {device}')

# 2. 准备数据
from google.colab import drive
drive.mount('/content/drive')

# 3. 克隆代码
!git clone https://github.com/your-username/MSHNet.git
%cd MSHNet

# 4. 安装依赖
!pip install -q torch-xla cloud-tpu-client scikit-image tqdm

# 5. 开始训练
!python main_tpu.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 32 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --use-tpu
```

### 多核TPU训练

```python
# 使用8个TPU核心并行训练
!python main_tpu.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 32 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --use-tpu \
    --num-cores 8
```

---

## 🎯 最佳实践

### 1. 训练前准备

```python
# ✅ 检查TPU状态
import torch_xla.core.xla_model as xm
print(f'TPU设备: {xm.xla_device()}')
print(f'核心数: {xm.xrt_world_size()}')

# ✅ 验证数据集
!ls -lh datasets/IRSTD-1k/

# ✅ 小规模测试
!python main_tpu.py --epochs 2 --use-tpu  # 先跑2个epochs测试
```

### 2. 训练中监控

```python
# 查看XLA编译缓存
import torch_xla.debug.metrics as met
print(met.metrics_report())

# 监控内存使用
!nvidia-smi  # GPU
# TPU没有直接的内存监控工具
```

### 3. 训练后处理

```python
# 保存结果到Drive
!cp -r ./weight/* /content/drive/MyDrive/MSHNet_results/

# 打包权重
!tar -czf tpu_weights.tar.gz ./weight/
!cp tpu_weights.tar.gz /content/drive/MyDrive/
```

---

## 📞 支持资源

- **PyTorch XLA文档**: https://pytorch.org/xla/
- **Colab TPU教程**: https://colab.research.google.com/notebooks/tpu.ipynb
- **XLA性能指南**: https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md

---

**祝TPU训练顺利！🚀**

