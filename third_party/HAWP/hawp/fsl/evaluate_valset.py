import os
import json
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import numpy as np

from hawp.base.utils.logger import setup_logger
from hawp.fsl.dataset.build import build_test_dataset
from hawp.fsl.model import build_model
from hawp.fsl.config import cfg


# ============================================================
# 🧩 工具函数
# ============================================================
def _unwrap_to_dataset(ds):
    from torch.utils.data import DataLoader
    if isinstance(ds, (list, tuple)) and len(ds) == 2 and isinstance(ds[1], DataLoader):
        ds = ds[1]
    while isinstance(ds, DataLoader):
        ds = ds.dataset
    return ds


def _make_safe_loader(dataset, batch_size=1, num_workers=0):
    from torch.utils.data import DataLoader

    base = _unwrap_to_dataset(dataset)

    def _collate_identity(batch):
        if len(batch) == 1:
            return batch[0]
        elif isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            imgs, anns = zip(*batch)
            return list(imgs), list(anns)
        return batch

    return DataLoader(base, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=_collate_identity)


# ============================================================
# 🎨 可视化函数（带调试输出）
# ============================================================
def visualize_prediction(image_tensor, outputs, save_path, img_id, meta=None,
                         line_th=0.05, junc_th=0.05):
    os.makedirs(save_path, exist_ok=True)

    # --- 反归一化 ---
    mean = torch.tensor([109.73, 103.832, 98.681]) / 255.0
    std = torch.tensor([22.275, 22.124, 23.229]) / 255.0
    img = image_tensor.clone().cpu()
    img = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    img_pil = F.to_pil_image(img.squeeze(0))
    draw = ImageDraw.Draw(img_pil)

    # --- 原图大小 ---
    if meta is not None:
        H, W = meta.get("height", img_pil.height), meta.get("width", img_pil.width)
    else:
        H, W = img_pil.height, img_pil.width

    # --- 自动缩放因子修正 ---
    # 模型输出坐标通常已是原图尺度(512×512)，但部分旧模型仍在128域。
    # 若最大坐标小于原图尺寸的 1.5 倍，则自动放大，否则直接使用1.0。
    max_coord = 0
    if "lines_pred" in outputs and len(outputs["lines_pred"]) > 0:
        arr = outputs["lines_pred"]
        if torch.is_tensor(arr):  # 强制转为 numpy
            arr = arr.detach().cpu().numpy()
        max_coord = float(np.max(arr))
    elif "juncs_pred" in outputs and len(outputs["juncs_pred"]) > 0:
        arr = outputs["juncs_pred"]
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        max_coord = float(np.max(arr))

    scale_x, scale_y = 1.0, 1.0

    # ============================================================
    # 🧭 调试输出 1：坐标与分数范围
    # ============================================================
    if "lines_pred" in outputs:
        lp = outputs["lines_pred"]
        ls = outputs.get("lines_score", torch.ones(len(lp)))
        if torch.is_tensor(lp):
            lp = lp.detach().cpu().numpy()
            ls = ls.detach().cpu().numpy()
        if len(lp) > 0:
            print(f"[{img_id}] lines_pred shape={np.array(lp).shape}, "
                  f"coord_range=({lp.min():.2f}, {lp.max():.2f}), "
                  f"score_range=({ls.min():.3f}, {ls.max():.3f}), mean={ls.mean():.3f}")
        else:
            print(f"[{img_id}] ⚠️ lines_pred empty")

    if "juncs_pred" in outputs:
        jp = outputs["juncs_pred"]
        js = outputs.get("juncs_score", torch.ones(len(jp)))
        if torch.is_tensor(jp):
            jp = jp.detach().cpu().numpy()
            js = js.detach().cpu().numpy()
        if len(jp) > 0:
            print(f"[{img_id}] juncs_pred shape={np.array(jp).shape}, "
                  f"coord_range=({jp.min():.2f}, {jp.max():.2f}), "
                  f"score_range=({js.min():.3f}, {js.max():.3f}), mean={js.mean():.3f}")
        else:
            print(f"[{img_id}] ⚠️ juncs_pred empty")

    # ============================================================
    # 绘制 junctions（支持负阈值=关闭过滤；带统计）
    # ============================================================
    keep_j = 0
    if "juncs_pred" in outputs:
        # jp/js 在前面的调试输出里已经转换为 numpy；若没有则兜底
        if "jp" not in locals() or "js" not in locals():
            jp = outputs["juncs_pred"]
            js = outputs.get("juncs_score", torch.ones(len(jp)))
            if torch.is_tensor(jp): jp = jp.detach().cpu().numpy()
            if torch.is_tensor(js): js = js.detach().cpu().numpy()
        total_j = len(jp)
        for (x, y), s in zip(jp, js):
            # 负阈值 => 不过滤
            if junc_th >= 0 and s < junc_th:
                continue
            keep_j += 1
            r = 2 + int(3 * float(s))
            color = (0, int(255 * float(s)), 0)  # 分数越高越亮
            draw.ellipse([(x * scale_x - r, y * scale_y - r),
                          (x * scale_x + r, y * scale_y + r)],
                         outline=color, width=1)
        print(f"[{img_id}] junctions kept: {keep_j}/{total_j} (th={junc_th})")

    # ============================================================
    # 绘制 lines（支持负阈值=关闭过滤；带统计与颜色映射）
    # ============================================================
    keep_l = 0
    if "lines_pred" in outputs:
        # lp/ls 在前面的调试输出里已经转换为 numpy；若没有则兜底
        if "lp" not in locals() or "ls" not in locals():
            lp = outputs["lines_pred"]
            ls = outputs.get("lines_score", torch.ones(len(lp)))
            if torch.is_tensor(lp): lp = lp.detach().cpu().numpy()
            if torch.is_tensor(ls): ls = ls.detach().cpu().numpy()
        # 兜底：若没有分数或长度不匹配，统一置为 1
        if ls is None or (hasattr(ls, "__len__") and len(ls) != len(lp)):
            ls = np.ones((len(lp),), dtype=np.float32)
    
        total_l = len(lp)
        for (x1, y1, x2, y2), s in zip(lp, ls):
            # 负阈值 => 不过滤
            if line_th >= 0 and s < line_th:
                continue
            keep_l += 1
            # 给分数一个直观色彩：高分更红，低分偏黄
            s_float = float(s)
            color = (255, int(255 * (1 - s_float)), 0)
            draw.line(
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                fill=color, width=max(1, int(2 + 2 * s_float))
            )
        print(f"[{img_id}] lines kept: {keep_l}/{total_l} (th={line_th})")

# ============================================================
# 🚀 验证主逻辑
# ============================================================
@torch.no_grad()
def run_validation(model, dataset, device, save_path,
                   line_th=0.05, junc_th=0.05):
    model.eval()
    results = []
    total_refine = 0.0
    total_batches = 0

    vis_dir = os.path.join(save_path, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    data_loader = _make_safe_loader(dataset, batch_size=1, num_workers=0)

    for i, sample in enumerate(tqdm(data_loader, desc="Validating")):
        if isinstance(sample, dict):
            images = sample.get("image", None)
            annotations = sample
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            images, annotations = sample[0], sample[1]
        else:
            continue

        # 图像加载
        if isinstance(images, str) and os.path.exists(images):
            img = Image.open(images).convert("RGB")
            images = F.to_tensor(img).unsqueeze(0).to(device)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0).to(device)
            else:
                images = images.to(device)
        elif isinstance(images, list) and isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0).to(device)
        else:
            continue
  
        if isinstance(annotations, dict):
            annotations = [annotations]

        try:
            outputs, extra_info = model(images)  # 推理：不要传 annotations
        except Exception as e:
            print(f"[❌ ERROR] 推理失败: {e}")
            continue

        if isinstance(extra_info, dict) and "valid_refine" in extra_info:
            total_refine += float(extra_info["valid_refine"])
        total_batches += 1

        img_id = "unknown"
        if isinstance(annotations[0], dict):
            fn = annotations[0].get("filename", None)
            if isinstance(fn, str):
                img_id = fn
            elif isinstance(fn, (list, tuple)) and len(fn) > 0:
                img_id = fn[0]

        entry = {"image_id": img_id}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if hasattr(v, "detach"):
                    entry[k] = v.detach().cpu().tolist()
        results.append(entry)

        # 🎨 可视化 & 调试
        visualize_prediction(images, outputs, vis_dir,
                             os.path.basename(str(img_id)),
                             annotations[0], line_th, junc_th)

    avg_refine = total_refine / max(total_batches, 1)
    print(f"\n✅ Validation Done: {total_batches} samples processed.")
    print(f"Average valid_refine = {avg_refine:.4f}")

    out_file = os.path.join(save_path, "hawp_val_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_file}")

    return avg_refine


# ============================================================
# 🏁 主入口（含配置打印）
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HAWP Validation Script (Debug Enhanced)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--line-th", type=float, default=0.05)
    parser.add_argument("--junc-th", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger("hawp.val", args.output)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")

    cfg.merge_from_file(args.config)
    cfg.defrost()
    if not hasattr(cfg.DATASETS, "VAL") or len(cfg.DATASETS.VAL) == 0:
        cfg.DATASETS.VAL = ["custom_val"]
    cfg.freeze()

    print("⚙️ Config Check:")
    print(f"  - IMAGE SIZE: {cfg.DATASETS.IMAGE.HEIGHT}x{cfg.DATASETS.IMAGE.WIDTH}")
    print(f"  - PIXEL_MEAN: {cfg.DATASETS.IMAGE.PIXEL_MEAN}")
    print(f"  - PIXEL_STD : {cfg.DATASETS.IMAGE.PIXEL_STD}")
    print(f"  - DIST_TH   : {getattr(cfg.DATASETS, 'DISTANCE_TH', 'N/A')}")
    print(f"  - AUGMENTATION: {getattr(cfg.DATASETS, 'AUGMENTATION', 'N/A')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    logger.info("Checkpoint loaded successfully.")
    # ✅ 让模型使用命令行指定的 line-th 阈值
    model._manual_conf_th = max(args.line_th, 0.01)
    model._manual_junc_th = max(args.junc_th, 0.01)

    dataset = build_test_dataset(cfg)
    from torch.utils.data import DataLoader
    if isinstance(dataset, list) and len(dataset) == 1 and isinstance(dataset[0], (tuple, list)):
        name, inner = dataset[0]
        print(f"⚙️ 自动展开 list[({name}, DataLoader)] → inner.dataset")
        dataset = inner.dataset if isinstance(inner, DataLoader) else inner
    print(f"✅ 最终 dataset 类型: {type(dataset)} | 长度: {len(dataset) if hasattr(dataset,'__len__') else 'unknown'}")

    avg_refine = run_validation(model, dataset, device, args.output,
                                line_th=args.line_th, junc_th=args.junc_th)
    logger.info(f"Validation Finished. Average valid_refine = {avg_refine:.4f}")

    print(f"📄 Summary written to {os.path.join(args.output, 'val_summary.txt')}")


if __name__ == "__main__":
    main()
