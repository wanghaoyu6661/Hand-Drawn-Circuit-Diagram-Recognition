# -*- coding: utf-8 -*-

import os
import json
import torch
import yaml
import math
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import sys
from path_config import cfg_get
sys.path.append(cfg_get("legacy", "parseq_repo", default="/root/autodl-tmp/parseq-main"))


def load_parseq_model(ckpt_path, charset_path, device):
    print(f"🔹 Loading PARSeq from {ckpt_path}")

    model = torch.hub.load(
        cfg_get("legacy", "parseq_torchhub_repo", default="/root/.cache/torch/hub/baudm_parseq_main"),
        'parseq',
        source='local',
        pretrained=False
    )

    with open(charset_path, "r", encoding="utf-8") as f:
        charset = [line.strip() for line in f if line.strip()]

    from strhub.data.utils import Tokenizer
    model.tokenizer = Tokenizer(charset)

    num_classes = len(charset) + 3
    in_features = model.model.head.in_features
    model.model.head = torch.nn.Linear(in_features, num_classes)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()
    print("✅ Model loaded successfully!")
    return model


def euclidean_distance_xy(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def fuse_per_image(image_name, image_path, label_path, model, device, output_dir, component_names):
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not os.path.exists(label_path):
        print(f"[WARN] Missing label for {image_name}")
        return

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    with open(label_path, "r") as f:
        lines = [x.strip().split() for x in f.readlines() if len(x.strip().split()) >= 6]

    text_boxes = []
    comp_boxes = []

    # box format: [xc, yc, bw, bh, conf, cls_id, line_idx]
    for i, l in enumerate(lines):
        cls, xc, yc, bw, bh, conf = map(float, l)
        box = [xc, yc, bw, bh, conf, int(cls), i]
        if int(cls) == 0:
            text_boxes.append(box)
        elif int(cls) not in [0, 1, 2]:  # 排除 text/junction/crossover
            comp_boxes.append(box)

    results = []

    for tb in tqdm(text_boxes, desc=f"[{image_name}] OCR中"):
        xc, yc, bw, bh, conf, cls_id, idx = tb

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        # --- 自适应 padding：按文本框尺寸留白，减少切字/笔画被截断 ---
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad = 0.15 * max(box_w, box_h)        # 15% 边距（你可调：0.10~0.25）
        pad = max(2.0, min(pad, 20.0))        # clamp：避免小框 padding 太小 / 大框太大

        x1p = max(0.0, x1 - pad)
        y1p = max(0.0, y1 - pad)
        x2p = min(float(w), x2 + pad)
        y2p = min(float(h), y2 + pad)

        # 防御：避免极端情况下出现空框
        if x2p - x1p < 2 or y2p - y1p < 2:
            x1p, y1p, x2p, y2p = x1, y1, x2, y2

        crop = img.crop((x1p, y1p, x2p, y2p))

        crop_t = transform(crop).unsqueeze(0).to(device)
        try:
            probs = model(crop_t).softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            pred_text = preds[0] if preds else ""
        except Exception as e:
            print(f"  ⚠️ OCR failed on {image_name}: {e}")
            pred_text = ""

        # -------- 类别级近邻匹配（只输出 cls_id，不输出实例 id）--------
        min_dist = float("inf")
        nearest_cls_id = None

        for comp in comp_boxes:
            c_x, c_y = comp[0], comp[1]
            dist = euclidean_distance_xy(xc, yc, c_x, c_y)
            if dist < min_dist:
                min_dist = dist
                nearest_cls_id = int(comp[5])  # 这里是 cls_id（类别号）

        # --- 自适应距离阈值：随文本框大小变化（bw,bh 是 YOLO normalized）---
        text_scale = math.sqrt(bw * bw + bh * bh)      # 文本框对角线（归一化）
        dist_th = 0.10 + 0.90 * text_scale             # 小框≈0.10，大框允许更远
        dist_th = max(0.10, min(dist_th, 0.22))        # clamp：避免过松/过严
        #print(f"[DBG@{image_name}] text_scale={text_scale:.4f} dist={min_dist:.4f} th={dist_th:.4f} pad={pad:.1f}")

        relation = "component" if min_dist < dist_th else "global_description"

        cls_name = None
        if relation == "component" and nearest_cls_id is not None and 0 <= nearest_cls_id < len(component_names):
            cls_name = component_names[nearest_cls_id]

        results.append({
            "image": image_name,
            "text": pred_text,
            "text_box": [xc, yc, bw, bh],          # normalized xywh
            "match_relation": relation,
            "distance": float(min_dist),

            # ✅ 类别级输出（可靠、语义正确）
            "matched_component_cls_id": int(nearest_cls_id) if relation == "component" else None,
            "component_cls_name": cls_name if relation == "component" else None,

            # 🚫 实例级 id 在此阶段不输出（避免误用）
            "matched_component_id": None,
        })

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{image_name}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {image_name}.json ({len(results)} entries)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_label_dir = cfg_get("paths", "yolo_labels", default="/root/autodl-tmp/final_result/yolo_detect/exp/labels")
    src_img_dir = cfg_get("paths", "src_img", default="/root/autodl-tmp/final_result/src_img")
    output_dir = cfg_get("paths", "fuse_json", default="/root/autodl-tmp/final_result/fuse_json")
    ckpt = cfg_get("weights", "parseq_ckpt", default="/root/autodl-tmp/handocr_project/output_parseq/best_parseq.pt")
    charset_path = cfg_get("weights", "parseq_charset", default="/root/autodl-tmp/handwritten_ocr/scripts/my_dict.txt")
    data_yaml = (
        cfg_get("weights", "yolo_data_yaml", default=None)
        or cfg_get("paths", "yolo_data_yaml", default=None)
        or "/root/yolov9-main/data/data.yaml"
    )

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    component_names = data['names']

    model = load_parseq_model(ckpt, charset_path, device)

    label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith(".txt")]

    for lbl in label_files:
        base_name = os.path.splitext(lbl)[0]
        img_path = os.path.join(src_img_dir, f"{base_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(src_img_dir, f"{base_name}.jpg")

        label_path = os.path.join(yolo_label_dir, lbl)
        if os.path.exists(img_path):
            fuse_per_image(base_name, img_path, label_path, model, device, output_dir, component_names)
        else:
            print(f"[WARN] Missing image for {base_name}")


if __name__ == "__main__":
    main()
