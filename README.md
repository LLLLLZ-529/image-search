# 🖼️ Image Search by Image

基于 **DINOv2（纯 NumPy 实现）** 的以图搜图 Web 应用：上传一张图片，从图库中检索出最相似的 Top-K 结果。

本项目是深度学习课程作业/实验项目：不依赖 PyTorch 推理框架，用 **NumPy 从零实现 ViT-DINOv2** 的前向过程（Attention / MLP / LayerNorm / GELU），再通过余弦相似度做图库检索。

## ✨ 功能特性

- 🔍 **以图搜图**：上传图片 → 提取 DINOv2 特征 → 与图库特征计算余弦相似度 → 返回 Top-K
- 🧠 **纯 NumPy 实现 ViT**：`dinov2_numpy.py` 完整实现多头注意力、MLP、LayerNorm 等模块
- 🖥️ **Flask Web UI**：上传页面 + 结果展示（`templates/`、`static/`）
- 🗂️ **图库特征预计算**：图库图片特征存为 `.npy`，元数据存 `.jsonl`
- 📏 **图片预处理**：`resize_short_side` 短边缩放 + 14 倍数对齐（适配 ViT patch 尺寸）

## 🚀 快速开始

### 环境依赖

```bash
pip install flask numpy scipy pillow
```

### 1. 准备模型权重与图库

- 模型权重：`vit-dinov2-base.npz`（代码从该文件加载 ViT 权重）【待确认：权重来源与下载方式】
- 图库特征：`gallery/gallery_features.npy` + `gallery/gallery_metadata.jsonl`（每行一条元数据，含 image_path / image_url / caption）

### 2. 校验特征提取是否正确（作业 debug 步骤）

```bash
# 运行特征提取，与参考特征 demo_data/cat_dog_feature.npy 对比，误差应在数值容差内
python -c "from dinov2_numpy import Dinov2Numpy; ..."   # 具体以你的 debug 脚本为准
```

### 3. 启动

```bash
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

### 4. 检索

上传图片后，服务端会：
1. `resize_short_side` 预处理到 224×224
2. DINOv2 提取查询特征
3. 与图库特征点积（归一化后即余弦相似度）
4. 返回 Top-K 结果（含分数、图片路径、说明文字）

## 📁 项目结构

```
Image-Search-by-Image/
├── app.py               # Flask 服务：检索接口与页面
├── dinov2_numpy.py      # DINOv2 纯 NumPy 实现（模型前向）
├── preprocess_image.py  # resize_short_side 等预处理
├── demo_data/           # 演示图片与参考特征（cat.jpg / dog.jpg / cat_dog_feature.npy）
├── static/              # 前端样式与脚本
├── templates/           # index.html 页面
└── readme.txt           # 原始作业说明（已整理进本 README）
```

## 📝 项目说明（源自作业要求）

1. 用纯 NumPy 完成 DINOv2 前向实现，并与参考特征比对验证正确性；
2. 图库规模目标 10,000+ 张网络图片（`data.csv` 收集），批量提取特征；
3. 检索用余弦相似度（或 L2 距离），返回 Top-10。

## ⚠️ 待确认事项

- `vit-dinov2-base.npz` 权重的获取方式（官方权重转换脚本？）——README 定稿前请补充
- `preprocess_image.py` 的 `resize_short_side` 是否已按作业要求实现完成（代码中该函数是关键，建议跑通 demo 验证）

## 📄 许可

未指定开源许可（默认保留所有权利）。
