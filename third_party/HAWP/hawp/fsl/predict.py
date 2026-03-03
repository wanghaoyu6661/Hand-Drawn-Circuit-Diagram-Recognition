import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from hawp.base.utils.logger import setup_logger
from hawp.fsl.config import cfg
from hawp.fsl.dataset.build import build_test_dataset
from hawp.fsl.model.build import build_model


# ---------------------------------------------------------
# 🔧 工具函数
# ---------------------------------------------------------
def set_random_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _resolve_dataset(obj):
    """
    安全解包 HAWP 返回的多层嵌套 dataset:
    - [("name", DataLoader)]
    - ("name", DataLoader)
    - DataLoader
    - dataset
    """
    from torch.utils.data import DataLoader

    # 情况1: list 包裹 ("name", DataLoader)
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], (tuple, list)) and len(obj[0]) == 2:
        name, inner = obj[0]
        if isinstance(inner, DataLoader):
            print(f"⚙️ 自动展开 list[({name}, DataLoader)] → inner.dataset")
            return inner.dataset
        return inner

    # 情况2: ("name", DataLoader)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        name, inner = obj
        if isinstance(inner, DataLoader):
            print(f"⚙️ 自动展开 ({name}, DataLoader) → inner.dataset")
            return inner.dataset
        return inner

    # 情况3: DataLoader
    if isinstance(obj, DataLoader):
        print("⚙️ 自动展开 DataLoader → inner.dataset")
        return obj.dataset

    # 默认：原样返回
    return obj


# ---------------------------------------------------------
# 🚀 验证推理函数
# ---------------------------------------------------------
@torch.no_grad()
def run_validation(model, dataset, device, output_dir):
    """
    统一验证逻辑（稳定版）
    - 自动处理 DataLoader/dataset 结构
    - 自动保存 JSON 输出
    """
    from PIL import Image
    import torchvision.transforms.functional as F
    from torch.utils.data import DataLoader

    model.eval()
    results = []
    total_refine = 0.0
    total_batches = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(tqdm(loader, desc="Validating")):
        if isinstance(sample, dict):
            images = sample.get("image", None)
            annotations = sample
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            images, annotations = sample[0], sample[1]
        else:
            print(f"[⚠️ Skip] Unexpected sample type: {type(sample)}")
            continue

        # 图像加载
        if isinstance(images, str) and os.path.exists(images):
            img = Image.open(images).convert("RGB")
            images = F.to_tensor(img).unsqueeze(0).to(device)
        elif hasattr(images, "to"):
            images = images.to(device)
        else:
            print(f"[⚠️ Skip] 非法图像类型: {type(images)}")
            continue

        # 注解上 GPU
        if isinstance(annotations, dict):
            annotations = {k: v.to(device) if hasattr(v, "to") else v for k, v in annotations.items()}

        # 前向推理
        outputs, extra_info = model(images, annotations)

        if isinstance(extra_info, dict) and "valid_refine" in extra_info:
            total_refine += float(extra_info["valid_refine"])
        total_batches += 1

        # 保存结果
        img_id = "unknown"
        if isinstance(annotations, dict):
            fn = annotations.get("filename", None)
            if isinstance(fn, (list, tuple)) and len(fn) > 0:
                img_id = fn[0]
            elif isinstance(fn, str):
                img_id = fn

        entry = {"image_id": img_id}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if hasattr(v, "detach"):
                    entry[k] = v.detach().cpu().tolist()
        results.append(entry)

    # 汇总结果
    avg_refine = total_refine / max(total_batches, 1)
    print(f"\n✅ Validation Done: {total_batches} samples processed.")
    print(f"Average valid_refine = {avg_refine:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "hawp_val_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_file}")

    return avg_refine


# ---------------------------------------------------------
# 🎯 主函数
# ---------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HAWP Validation Script (stable unified style)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger("hawp.val", args.output)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")

    # --------------------
    # 加载配置与随机种子
    # --------------------
    cfg.merge_from_file(args.config)
    cfg.defrost()
    if not hasattr(cfg.DATASETS, "VAL") or len(cfg.DATASETS.VAL) == 0:
        cfg.DATASETS.VAL = ["custom_val"]
    cfg.freeze()
    set_random_seed(args.seed)

    # --------------------
    # 模型加载
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    logger.info("Checkpoint loaded successfully.")

    # --------------------
    # 构建验证集
    # --------------------
    from hawp.fsl.config.paths_catalog import DatasetCatalog
    try:
        dataset = build_test_dataset(cfg)
    except FileNotFoundError as e:
        msg = str(e)
        if "data_hawp_last" in msg:
            candidates = [
                "/root/autodl-tmp/HAWP/data/data_hawp_last/json_converted/val_hawp_style.json",
                "/root/autodl-tmp/data_hawp_last/json_converted/val_hawp_style.json",
            ]
            for fixed in candidates:
                if os.path.exists(fixed):
                    print(f"⚙️ 自动修正 ann_file 路径为: {fixed}")
                    if "custom_val" in DatasetCatalog.DATASETS:
                        DatasetCatalog.DATASETS["custom_val"]["ann_file"] = os.path.relpath(
                            fixed, DatasetCatalog.DATA_DIR
                        )
                    dataset = build_test_dataset(cfg)
                    break
            else:
                raise e
        else:
            raise e

    # --------------------
    # 打印结构调试
    # --------------------
    print("\n================ DEBUG: build_test_dataset(cfg) 返回结构 ================")
    print(f"type(dataset): {type(dataset)}")
    if isinstance(dataset, (list, tuple)):
        print(f"len(dataset): {len(dataset)}")
        for i, x in enumerate(dataset):
            print(f"  ├─ [{i}] type={type(x)} | len={len(x) if hasattr(x, '__len__') else 'NA'}")
    print("====================================================================\n")

    # --------------------
    # 自动展开 dataset
    # --------------------
    dataset = _resolve_dataset(dataset)
    logger.info(f"Loaded validation dataset: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples.")

    # --------------------
    # 运行验证
    # --------------------
    avg_refine = run_validation(model, dataset, device, args.output)
    logger.info(f"Validation Finished. Average valid_refine = {avg_refine:.4f}")

    with open(os.path.join(args.output, "val_summary.txt"), "w") as f:
        f.write(f"Average valid_refine = {avg_refine:.4f}\n")
    print(f"📄 Summary written to {os.path.join(args.output, 'val_summary.txt')}")


if __name__ == "__main__":
    main()
