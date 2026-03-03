# -*- coding: utf-8 -*-

import os
import cv2
import glob
import numpy as np
from path_config import cfg_get

# ===============================
#   与 remove_components.py 完全同步
# ===============================
EXPAND_PIXELS_INTEGRATED_CIRCUIT = 0   # 对类 33/34/35
EXPAND_PIXELS_OTHER = 0                # 对其它所有类

IC_CLASSES = [33, 34, 35]              # 集成电路类 ID
JUNC_CLASS = 1                         # junction 类 ID（不遮挡）

# 你的灰色值
GRAY_COLOR = (128, 123, 130)

# ===============================
# 工具函数
# ===============================

def xywhn_to_xyxy(x, y, w, h, W, H):
    x1 = int((x - w/2) * W)
    y1 = int((y - h/2) * H)
    x2 = int((x + w/2) * W)
    y2 = int((y + h/2) * H)
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, expand_pixels, img_w, img_h):
    """同步 remove_components.py 中逻辑"""
    x1_exp = max(0, x1 - expand_pixels)
    y1_exp = max(0, y1 - expand_pixels)
    x2_exp = min(img_w, x2 + expand_pixels)
    y2_exp = min(img_h, y2 + expand_pixels)
    return x1_exp, y1_exp, x2_exp, y2_exp


def load_yolo_boxes_with_expand(label_path, W, H):
    """读取 YOLO 的目标框，同时按 remove_components.py 的逻辑进行 expand"""
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            p = line.strip().split()
            cls = int(p[0])
            if cls == JUNC_CLASS:
                continue

            x, y, bw, bh = map(float, p[1:5])
            x1, y1, x2, y2 = xywhn_to_xyxy(x, y, bw, bh, W, H)

            # === 同步 remove_components.py 逻辑 ===
            if cls in IC_CLASSES:
                x1, y1, x2, y2 = expand_bbox(
                    x1, y1, x2, y2,
                    EXPAND_PIXELS_INTEGRATED_CIRCUIT,
                    W, H
                )
            else:
                x1, y1, x2, y2 = expand_bbox(
                    x1, y1, x2, y2,
                    EXPAND_PIXELS_OTHER,
                    W, H
                )

            boxes.append((x1, y1, x2, y2))

    return boxes


# ===============================
# 主流程
# ===============================

def main():
    IMG_DIR = cfg_get("paths", "src_img", default="/root/autodl-tmp/final_result/src_img")
    LABEL_DIR = cfg_get("paths", "yolo_labels", default="/root/autodl-tmp/final_result/yolo_detect/exp/labels")
    OUT_DIR = cfg_get("paths", "suppressed_img", default="/root/autodl-tmp/final_result/lama_clean")
    os.makedirs(OUT_DIR, exist_ok=True)

    img_paths = glob.glob(os.path.join(IMG_DIR, "*"))
    print(f"🔍 共发现 {len(img_paths)} 张图像")

    for img_path in img_paths:
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        # 读取并扩展 bounding boxes
        boxes = load_yolo_boxes_with_expand(label_path, W, H)

        # === 用灰色覆盖所有 bounding boxes ===
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), GRAY_COLOR, -1)

        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
        print("✅ 保存：", out_path)


if __name__ == "__main__":
    main()
