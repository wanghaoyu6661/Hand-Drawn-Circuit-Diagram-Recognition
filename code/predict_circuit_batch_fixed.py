import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from hawp.base.utils.logger import setup_logger
from hawp.fsl.config import cfg
from hawp.fsl.model import build_model
from hawp.fsl.dataset.build import build_transform


# ============================================================
# 🎨 只画点的可视化函数
# ============================================================
def visualize_junctions_only(img_path, outputs, save_dir, img_name,
                             junc_th=0.05, font_size=14):

    os.makedirs(save_dir, exist_ok=True)
    img_pil = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    jp = outputs["junctions_pred"]
    js = outputs["junctions_score"]

    # JSON 已经过滤，这里不再过滤
    for (x, y), s in zip(jp, js):
        r = 2 + int(3 * s)
        color = (int(255 * (1 - s)), int(255 * s), 0)

        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     outline=color, width=2)
        draw.text((x + 4, y - 8),
                  f"{s:.2f}", fill=color, font=font)

    save_path = os.path.join(save_dir, f"{img_name}_junctions.png")
    img_pil.save(save_path)
    print(f"[✅] 保存点可视化: {save_path}")



# ============================================================
# ⚙️ 只推理 Junction 的函数
# ============================================================
@torch.no_grad()
def run_inference_junc_only(model, transform, img_dir, device, save_dir,
                            junc_th=0.05):

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visuals")
    json_dir = os.path.join(save_dir, "json")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    img_list = [f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    print(f"🖼️ 共发现 {len(img_list)} 张图像。")

    for fname in tqdm(img_list, desc="Predicting"):
        img_path = os.path.join(img_dir, fname)
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except:
            continue

        img_np = np.array(img_pil)
        img_tensor = transform(img_np).unsqueeze(0).to(device)

        width, height = img_pil.size

        # -----------------------------
        # 🔥 只用 forward_test → junctions_pred
        # -----------------------------
        outputs = model.forward_test(img_tensor, annotations=[{
            "filename": fname,
            "width": width,
            "height": height
        }])

        juncs_pred  = outputs.get("junctions_pred", None)
        juncs_score = outputs.get("junctions_score", None)

        if juncs_pred is None:
            juncs_pred  = torch.zeros((0, 2), device=device)
            juncs_score = torch.zeros((0,), device=device)

        # ✅ 调试：看一下分数大概分布
        if juncs_score.numel() > 0:
            print(f"[debug] {fname} #junc={len(juncs_score)} "
                f"min={juncs_score.min().item():.4f} "
                f"median={juncs_score.median().item():.4f} "
                f"max={juncs_score.max().item():.4f}")


        if juncs_score.numel() > 0:
            keep = juncs_score >= junc_th
            juncs_pred  = juncs_pred[keep]
            juncs_score = juncs_score[keep]


        out = {
            "filename": fname,
            "junctions_pred":  juncs_pred.cpu().tolist(),
            "junctions_score": juncs_score.cpu().tolist(),
        }

        # JSON 保存
        with open(os.path.join(json_dir, f"{os.path.splitext(fname)[0]}.json"), "w") as f:
            json.dump(out, f, indent=2)

        # 可视化保存
        visualize_junctions_only(
            img_path, out, vis_dir,
            os.path.splitext(fname)[0],
            junc_th=junc_th
        )

    print("\n🎉 完成，仅输出了 Junction")


# ============================================================
# 🏁 主入口：只推理 Junction
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HAWP Junction Only Inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--junc-th", type=float, default=0.04)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("hawp.predict_junction_only", args.output_dir)

    cfg.merge_from_file(args.config)
    cfg.freeze()

    transform = build_transform(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    print("✅ 模型加载完成。")
    run_inference_junc_only(model, transform,
                            args.input_dir, device, args.output_dir,
                            junc_th=args.junc_th)


if __name__ == "__main__":
    main()
