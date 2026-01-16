import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

app = Flask(__name__)

# 配置
GALLERY_DIR = Path("gallery")
FEATURES_PATH = GALLERY_DIR / "gallery_features.npy"
METADATA_PATH = GALLERY_DIR / "gallery_metadata.jsonl"
WEIGHTS_PATH = "vit-dinov2-base.npz"
UPLOAD_FOLDER = "temp_uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局加载模型和数据
print("Loading model and gallery data...")
try:
    # 1. 加载模型
    vit_model = Dinov2Numpy(WEIGHTS_PATH)
    
    # 2. 加载特征库
    if FEATURES_PATH.exists():
        gallery_features = np.load(FEATURES_PATH)
        # 归一化特征库以便计算余弦相似度
        norms = np.linalg.norm(gallery_features, axis=1, keepdims=True)
        gallery_features = gallery_features / (norms + 1e-6)
    else:
        gallery_features = None
        print(f"Warning: {FEATURES_PATH} not found.")

    # 3. 加载元数据
    gallery_metadata = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gallery_metadata.append(json.loads(line))
    else:
        print(f"Warning: {METADATA_PATH} not found.")

except Exception as e:
    print(f"Error during initialization: {e}")
    vit_model = None
    gallery_features = None
    gallery_metadata = []

print("Initialization complete.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        limit = int(request.form.get("limit", 5))
    except ValueError:
        limit = 5

    # 保存上传的图片用于处理
    temp_path = Path(UPLOAD_FOLDER) / file.filename
    file.save(temp_path)

    try:
        # 提取特征
        pixel_values = resize_short_side(str(temp_path), target_size=224)
        query_feature = vit_model(pixel_values).astype(np.float32).squeeze(0)
        
        # 归一化查询特征
        query_norm = np.linalg.norm(query_feature)
        query_feature = query_feature / (query_norm + 1e-6)

        if gallery_features is None or len(gallery_metadata) == 0:
            return jsonify({"error": "Gallery data not loaded"}), 500

        # 计算相似度 (Cosine Similarity = Dot product after normalization)
        # sim shape: (N,)
        similarities = np.dot(gallery_features, query_feature)
        
        # 获取 Top K 索引
        # 取更多候选，以防部分图片不存在
        candidate_count = min(limit * 4, len(similarities))
        top_indices = np.argsort(similarities)[-candidate_count:][::-1]
        
        results = []
        found_count = 0
        
        for idx in top_indices:
            if found_count >= limit:
                break
                
            score = float(similarities[idx])
            meta = gallery_metadata[idx]
            image_path_str = meta.get("image_path", "")
            
            # 检查文件是否存在
            # 兼容多种路径情况
            is_valid = False
            if image_path_str and os.path.exists(image_path_str):
                is_valid = True
            elif image_path_str and (GALLERY_DIR / Path(image_path_str).name).exists():
                # 尝试只用文件名在 gallery/images 下找? 
                # 实际上 image_path 应该是 "gallery/images/..."
                pass
            
            if not is_valid:
                continue

            results.append({
                "score": score,
                "image_path": image_path_str,
                "image_url": meta.get("image_url", ""),
                "caption": meta.get("caption", "")
            })
            found_count += 1

        return jsonify({"results": results})

    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时文件
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass


@app.route("/gallery/<path:filename>")
def serve_gallery_image(filename):
    # 提供 gallery 目录下的图片访问
    return send_from_directory(GALLERY_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
