# -*- coding: utf-8 -*-
"""
make_yolov10_crops_by_image.py (dynamic crop_size)

- 仍输出 320x320
- crop_size = round(CROP_SCALE * max(bbox_w, bbox_h))
- crop_size 会 clamp 到 [MIN_CROP, MAX_CROP]
- 文件名写入 _s{crop_size}，供 infer_port_vitpose.py 做正确的反算映射

输出目录结构不变：
  yolov10_crops/<image_stem>/<class_name>/*.jpg
"""

import os
from glob import glob
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import math
import re

# ====== 路径配置 ======
SRC_IMG_DIR = "/root/autodl-tmp/final_result/src_img_scanned"  # 你要用扫描风格的话，改成 src_img_scanned
YOLO_LABEL_DIR = "/root/autodl-tmp/final_result/yolo_detect/exp/labels"
CLASSES_TXT = "/root/autodl-fs/yolo_component_dataset/classes.txt"
OUT_DIR = "/root/autodl-tmp/final_result/yolov10_crops"

# ====== 输出与绘制参数 ======
GREEN_BOX_COLOR = (51, 255, 50)  # HCD 绿色框
GREEN_BOX_ALPHA = 150                 # ★可调：0~255，越小越透明（建议 80~140）
OUT_SIZE = 320                     # 最终输出固定 320x320
BOX_LINE_WIDTH = 5
BOX_PAD_INNER = 0                  # bbox 在 320 图上的内/外扩像素（正=外扩，负=内缩）

# ====== 动态 crop_size 参数（你只需要调这几个） ======
CROP_SCALE = 1.185185185   # ★可调：crop_size = CROP_SCALE * max(bw, bh)
MIN_CROP   = 10   # ★可调：最小 crop_size，避免小元件 crop 过大/过小
MAX_CROP   = 700   # ★可调：最大 crop_size，避免大元件 bbox 超出 crop

# 过滤阈值（如果 labels 带 conf）
CONF_THRES = 0.0


def load_classes(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float, Optional[float]]]:
    """
    返回 (cls_id, x, y, w, h, conf or None)
    """
    parts = line.strip().split()
    if len(parts) not in (5, 6):
        return None
    cls_id = int(float(parts[0]))
    x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
    conf = float(parts[5]) if len(parts) == 6 else None
    return cls_id, x, y, w, h, conf


def xywhn_to_xyxy_px(x, y, w, h, img_w, img_h):
    """
    归一化 xywh -> 像素 xyxy (float)
    """
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2


def clip_xyxy(x1, y1, x2, y2, img_w, img_h, pad=0):
    x1 = max(0, int(round(x1)) - pad)
    y1 = max(0, int(round(y1)) - pad)
    x2 = min(img_w, int(round(x2)) + pad)
    y2 = min(img_h, int(round(y2)) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_center_resize_draw_bbox(
    im: Image.Image,
    bbox_xyxy,             # bbox in original image pixel coords (int)
    out_path: str,
    crop_size: int,
    out_size: int = 320,
    box_color=(128, 253, 127),
    box_width: int = 5,
    box_pad_inner: int = 0,
    bg_color=(255, 255, 255),
):
    """
    按 bbox center 做正方形裁剪窗口 crop_size x crop_size，
    超出原图部分用 bg_color(白色) 填充，然后 resize 到 out_size，并画绿色 bbox。
    """
    W, H = im.size
    x1, y1, x2, y2 = bbox_xyxy

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    half = crop_size / 2.0
    tx1 = int(round(cx - half))
    ty1 = int(round(cy - half))
    tx2 = tx1 + crop_size
    ty2 = ty1 + crop_size

    canvas = Image.new("RGB", (crop_size, crop_size), bg_color)

    sx1 = max(0, tx1)
    sy1 = max(0, ty1)
    sx2 = min(W, tx2)
    sy2 = min(H, ty2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    patch = im.crop((sx1, sy1, sx2, sy2))

    px = sx1 - tx1
    py = sy1 - ty1
    canvas.paste(patch, (px, py))

    resized = canvas.resize((out_size, out_size), resample=Image.BILINEAR)

    # bbox 在 crop-local 坐标
    bx1 = x1 - tx1
    by1 = y1 - ty1
    bx2 = x2 - tx1
    by2 = y2 - ty1

    s = out_size / float(crop_size)
    rx1 = bx1 * s
    ry1 = by1 * s
    rx2 = bx2 * s
    ry2 = by2 * s

    rx1 -= box_pad_inner
    ry1 -= box_pad_inner
    rx2 += box_pad_inner
    ry2 += box_pad_inner

    rx1 = max(0, min(out_size - 1, rx1))
    ry1 = max(0, min(out_size - 1, ry1))
    rx2 = max(0, min(out_size - 1, rx2))
    ry2 = max(0, min(out_size - 1, ry2))

    # --- 用半透明 overlay 画框，避免遮住黑线细节 ---
    resized_rgba = resized.convert("RGBA")
    overlay = Image.new("RGBA", resized_rgba.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # 绿色带 alpha
    if len(box_color) == 3:
        rgba = (box_color[0], box_color[1], box_color[2], GREEN_BOX_ALPHA)
    else:
        rgba = box_color  # 如果你传进来本来就是 RGBA

    odraw.rectangle([rx1, ry1, rx2, ry2], outline=rgba, width=box_width)
    resized_rgba = Image.alpha_composite(resized_rgba, overlay).convert("RGB")
    resized_rgba.save(out_path, quality=95)

    return out_path

def compute_dynamic_crop_size(bw: int, bh: int) -> int:
    base = int(round(CROP_SCALE * float(max(bw, bh))))
    base = max(MIN_CROP, min(MAX_CROP, base))
    # 保证是偶数更稳（可选）
    if base % 2 == 1:
        base += 1
    return base


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = load_classes(CLASSES_TXT)
    print(f"[LOAD] classes: {len(classes)} from {CLASSES_TXT}")

    label_files = sorted(glob(os.path.join(YOLO_LABEL_DIR, "*.txt")))
    if not label_files:
        print("[WARN] No label files found:", YOLO_LABEL_DIR)
        return

    for lab_path in label_files:
        stem = os.path.splitext(os.path.basename(lab_path))[0]

        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            p = os.path.join(SRC_IMG_DIR, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            print(f"[SKIP] image not found for {stem} in {SRC_IMG_DIR}")
            continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        out_img_dir = os.path.join(OUT_DIR, stem)
        os.makedirs(out_img_dir, exist_ok=True)

        with open(lab_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            continue

        kept = 0
        for idx, ln in enumerate(lines, start=1):
            parsed = parse_label_line(ln)
            if parsed is None:
                continue
            cls_id, x, y, w, h, conf = parsed
            if cls_id < 0 or cls_id >= len(classes):
                continue
            if conf is not None and conf < CONF_THRES:
                continue

            cls_name = classes[cls_id]

            x1, y1, x2, y2 = xywhn_to_xyxy_px(x, y, w, h, W, H)
            xyxy = clip_xyxy(x1, y1, x2, y2, W, H, pad=0)
            if xyxy is None:
                continue
            x1i, y1i, x2i, y2i = xyxy

            bw = max(1, x2i - x1i)
            bh = max(1, y2i - y1i)
            crop_size = compute_dynamic_crop_size(bw, bh)

            out_cls_dir = os.path.join(out_img_dir, cls_name)
            os.makedirs(out_cls_dir, exist_ok=True)

            conf_str = f"{conf:.3f}" if conf is not None else "na"
            # ★关键：写入 _s{crop_size}
            fn = f"{idx:04d}_x{x1i}y{y1i}x{x2i}y{y2i}_s{crop_size}_c{conf_str}.jpg"
            out_path = os.path.join(out_cls_dir, fn)

            crop_center_resize_draw_bbox(
                im,
                (x1i, y1i, x2i, y2i),
                out_path,
                crop_size=crop_size,
                out_size=OUT_SIZE,
                box_color=GREEN_BOX_COLOR,
                box_width=BOX_LINE_WIDTH,
                box_pad_inner=BOX_PAD_INNER,
            )
            kept += 1

        print(f"[OK] {stem}: saved {kept} crops -> {out_img_dir}")


if __name__ == "__main__":
    main()
