# Google Colab A100 训练指南

本指南详细介绍如何在Google Colab的A100 GPU上训练MSHNet模型。

## 📋 目录
1. [准备工作](#准备工作)
2. [上传项目和数据](#上传项目和数据)
3. [运行训练](#运行训练)
4. [注意事项](#注意事项)
5. [常见问题](#常见问题)

---

## 🚀 准备工作

### 1. 获取Google Colab Pro（推荐）
- **免费版限制**：会话时长有限，可能中断训练
- **Colab Pro**：更长的会话时间，优先访问A100 GPU
- **Colab Pro+**：最长的会话时间和最高优先级

### 2. 准备数据集
将IRSTD-1k数据集上传到Google Drive：
```
Google Drive/
└── MSHNet_data/
    └── IRSTD-1k/
        ├── images/          # 或 IRSTD1k_Img/
        ├── masks/           # 或 IRSTD1k_Label/
        ├── trainval.txt
        ├── test.txt
        └── metafile.yaml
```

### 3. 准备项目代码
- 方法A：上传到Google Drive
- 方法B：上传到GitHub（推荐）
- 方法C：直接上传zip文件到Colab

---

## 📤 上传项目和数据

### 方法1：使用GitHub（推荐）

1. 将项目推送到GitHub
```bash
# 在本地执行
cd E:\Reserach\infrant-small-target\MSHNet
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/MSHNet.git
git push -u origin main
```

2. 在Colab中克隆
```python
!git clone https://github.com/your-username/MSHNet.git
%cd MSHNet
```

### 方法2：使用Google Drive

1. 上传项目文件夹到Drive
```
Google Drive/
└── MSHNet/
    ├── main.py
    ├── model/
    ├── utils/
    └── ...
```

2. 在Colab中挂载并复制
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/MSHNet /content/
%cd /content/MSHNet
```

---

## 🎯 运行训练

### 1. 打开Colab笔记本
- 访问：https://colab.research.google.com
- 上传 `colab_setup.ipynb` 文件
- 或创建新笔记本并复制代码

### 2. 选择A100 GPU
- 点击：**运行时** → **更改运行时类型**
- 硬件加速器：选择 **GPU**
- GPU类型：选择 **A100**（需要Colab Pro）

### 3. 按顺序运行单元格

#### 单元格1：检查GPU
```python
!nvidia-smi
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
```

#### 单元格2：安装依赖
```python
!pip install -q scikit-image tqdm
```

#### 单元格3：挂载Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 单元格4：准备代码和数据
```python
# 克隆代码
!git clone https://github.com/your-username/MSHNet.git
%cd MSHNet

# 链接数据集
!mkdir -p datasets
!ln -s /content/drive/MyDrive/MSHNet_data/IRSTD-1k ./datasets/IRSTD-1k
```

#### 单元格5：开始训练
```python
!python main.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 16 \
    --epochs 400 \
    --lr 0.05 \
    --mode train
```

---

## ⚠️ 注意事项

### A100优化参数

| 参数 | 本地值 | A100推荐值 | 说明 |
|-----|-------|-----------|-----|
| batch-size | 4 | 16-32 | A100显存40GB，可增大 |
| epochs | 400 | 400 | 保持不变 |
| lr | 0.05 | 0.05 | 保持不变 |
| base-size | 256 | 256 | 保持不变 |

### 训练时间估算

- **免费GPU**：约20-30小时（可能中断）
- **A100 GPU**：约8-12小时（连续训练）
- **建议**：使用Colab Pro确保不中断

### 定期保存策略

**重要**：Colab会话可能中断，必须定期保存！

#### 自动保存脚本
在训练中定期运行（每50 epochs）：
```python
# 在新单元格中运行
!cp -r /MSHNet/weight/* /content/drive/MyDrive/MSHNet_results/
print('✓ 权重已备份')
```

#### 使用checkpoint继续训练
如果训练中断，可以从checkpoint恢复：
```python
!python main.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 16 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --if-checkpoint True
```

### 监控训练进度

#### 方法1：实时查看日志
```python
# 在新单元格中运行
!tail -f /MSHNet/weight/MSHNet-*/metric.log
```

#### 方法2：查看最新IoU
```python
import glob
import os

weight_dirs = glob.glob('/MSHNet/weight/MSHNet-*')
if weight_dirs:
    latest_dir = max(weight_dirs, key=os.path.getctime)
    log_file = os.path.join(latest_dir, 'metric.log')
    if os.path.exists(log_file):
        !tail -10 {log_file}
```

---

## 🔧 常见问题

### Q1: 会话超时怎么办？
**A:** 
1. 使用Colab Pro获得更长会话时间
2. 设置自动点击脚本（JavaScript）：
```javascript
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

### Q2: 显存不足怎么办？
**A:** 
1. 减小batch-size：从16降到8或4
2. 减小图像尺寸：修改base-size和crop-size
3. 确认选择的是A100而不是T4

### Q3: 数据集目录结构不对？
**A:** 
检查代码中期望的目录结构：
```
datasets/IRSTD-1k/
├── images/      # 注意：不是IRSTD1k_Img
├── masks/       # 注意：不是IRSTD1k_Label
├── trainval.txt
└── test.txt
```

如果目录名不匹配，需要修改`utils/data.py`：
```python
# 第25-26行
self.imgs_dir = osp.join(dataset_dir, 'IRSTD1k_Img')   # 改为实际目录名
self.label_dir = osp.join(dataset_dir, 'IRSTD1k_Label') # 改为实际目录名
```

### Q4: 如何从中断处继续训练？
**A:** 
使用checkpoint功能：
```python
!python main.py \
    --dataset-dir './datasets/IRSTD-1k' \
    --batch-size 16 \
    --epochs 400 \
    --lr 0.05 \
    --mode train \
    --if-checkpoint True
```

### Q5: 训练完成后如何下载结果？
**A:** 
```python
# 打包所有权重
!tar -czf mshnet_weights.tar.gz /MSHNet/weight/

# 复制到Drive
!cp mshnet_weights.tar.gz /content/drive/MyDrive/

# 或直接下载到本地
from google.colab import files
files.download('mshnet_weights.tar.gz')
```

---

## 📊 性能对比

| GPU型号 | 显存 | 单epoch时间 | 总训练时间(400epochs) | Colab可用性 |
|---------|------|-------------|---------------------|------------|
| T4      | 16GB | ~3分钟      | ~20小时             | 免费版     |
| A100    | 40GB | ~1分钟      | ~7小时              | Pro/Pro+   |
| V100    | 16GB | ~2分钟      | ~13小时             | Pro        |

---

## 🎓 最佳实践

1. **训练前**：
   - ✅ 验证GPU是A100
   - ✅ 检查数据集完整性
   - ✅ 测试代码能正常运行

2. **训练中**：
   - ✅ 每50 epochs保存一次到Drive
   - ✅ 监控GPU使用率
   - ✅ 检查loss是否正常下降

3. **训练后**：
   - ✅ 保存所有权重到Drive
   - ✅ 运行测试验证性能
   - ✅ 下载结果到本地

---

## 📞 支持

如有问题，请参考：
- 项目README: `README.md`
- 论文: [Infrared Small Target Detection with Scale and Location Sensitivity](https://arxiv.org/abs/2403.19366)
- GitHub Issues

---

**祝训练顺利！🚀**

