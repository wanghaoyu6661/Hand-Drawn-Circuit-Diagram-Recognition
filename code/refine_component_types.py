# -*- coding: utf-8 -*-
"""
refine_component_types.py

放在 build_connections.py 之后、build_final_json.py 之前运行。

功能：
1) 细分 voltage.dc vs voltage.dc.one_port
   - 依据 link/json/*_graph.json 中 endpoints 对每个 voltage.dc 组件统计 terminal 数
2) 细分 transistor.bjt: npn/pnp 以及 transistor.fet: n/p
   - 使用 yolov10_crops/<base>/transistor.bjt/ & transistor.fet/
3) ✅ 新增：细分 DC source：V-DC vs I-DC
   - 先把 voltage.dc.one_port 剔除
   - 对 remaining two_port voltage.dc，用 DINOv2(dcsrc) 做二分类：vdc / idc
   - vdc -> cls_name_refined="voltage.dc"
   - idc -> cls_name_refined="current.dc"
"""

import os
import re
import json
import math
from glob import glob
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ============ 路径配置 ============
GRAPH_JSON_DIR = "/root/autodl-tmp/final_result/link/json"
YOLOV10_CROPS_DIR = "/root/autodl-tmp/final_result/yolov10_crops"

# ---- 你的权重（按你现有路径改）----
BJT_CKPT   = "/root/autodl-tmp/DINOv2/out_bjt_partial_ft/best_bjt_partial_ft.pt"
FET_CKPT   = "/root/autodl-tmp/DINOv2/out_mosfet_partial_ft/best_mosfet_partial_ft.pt"
DCSRC_CKPT = "/root/autodl-tmp/DINOv2/out_dcsrc_partial_ft/best_dcsrc_partial_ft.pt"   # ✅ 新增

OUT_DIR = "/root/autodl-tmp/final_result/type_refine/json"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "vit_large_patch14_dinov2"


# =======================
# utils
# =======================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_crops_for_image(cls_folder: str, image_stem: str) -> List[str]:
    """
    yolov10_crops/<image_stem>/<cls_folder>/*.jpg
    """
    folder = os.path.join(YOLOV10_CROPS_DIR, image_stem, cls_folder)
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files, key=natural_key)


def load_graph_json(path: str) -> Tuple[str, List[Dict], List[Dict]]:
    with open(path, "r") as f:
        js = json.load(f)
    image = js.get("image", "")
    components = js.get("components", [])
    endpoints = js.get("endpoints", [])
    return image, components, endpoints


# =======================
# DINOv2 分类器加载（兼容 linear / light-MLP head）
# =======================
class DinoClassifier(nn.Module):
    def __init__(self, backbone_name: str, head: nn.Module):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def _infer_head_from_state_dict(sd: Dict[str, torch.Tensor]) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    keys = list(sd.keys())

    # 情况1：MLP head（head.0 / head.2）
    if any(k.startswith("head.0.") for k in keys) and any(k.startswith("head.2.") for k in keys):
        w0 = sd["head.0.weight"]
        w2 = sd["head.2.weight"]

        in_dim = w0.shape[1]
        hid_dim = w0.shape[0]
        out_dim = w2.shape[0]

        head = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )
        head_sd = {k: v for k, v in sd.items() if k.startswith("head.")}
        return head, head_sd

    # 情况2：线性 head（head / classifier / fc）
    for prefix in ["head", "classifier", "fc"]:
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"
        if w_key in sd:
            w = sd[w_key]
            in_dim = w.shape[1]
            out_dim = w.shape[0]
            head = nn.Linear(in_dim, out_dim)
            head_sd = {"weight": sd[w_key]}
            if b_key in sd:
                head_sd["bias"] = sd[b_key]
            return head, head_sd

    raise RuntimeError("无法从 checkpoint 的 state_dict 推断分类头结构：未找到 head/classifier/fc 权重。")


def load_dino_classifier(ckpt_path: str) -> Tuple[nn.Module, List[str], Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd_full = ckpt["model"]
        classes = ckpt.get("classes", None)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd_full = ckpt["state_dict"]
        classes = ckpt.get("classes", None)
    elif isinstance(ckpt, dict):
        sd_full = ckpt
        classes = ckpt.get("classes", None)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    sd_full = {k.replace("module.", ""): v for k, v in sd_full.items()}

    head, head_sd = _infer_head_from_state_dict(sd_full)
    model = DinoClassifier(DINO_MODEL_NAME, head)

    # backbone：去掉 head/classifier/fc
    backbone_sd = {}
    for k, v in sd_full.items():
        if k.startswith("head.") or k.startswith("classifier.") or k.startswith("fc."):
            continue
        backbone_sd[k] = v

    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    model.head.load_state_dict(head_sd, strict=False)

    model.to(DEVICE).eval()

    # timm transform（尽量与训练一致）
    cfg = resolve_data_config({}, model=model.backbone)
    transform = create_transform(**cfg)

    if classes is None:
        # 没存 classes 就给个兜底（但你训练脚本里是会存的）
        num = list(model.head.parameters())[0].shape[0]
        classes = [f"cls{i}" for i in range(num)]

    return model, list(classes), transform


@torch.no_grad()
def predict_one(model: nn.Module, classes: List[str], transform, img_path: str) -> Tuple[str, float]:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(prob).item())
    return classes[idx], float(prob[idx].item())


# =======================
# 业务：VDC one-port 细分
# =======================
def refine_vdc_variant(components: List[Dict], endpoints: List[Dict]) -> Dict[int, Dict]:
    patch = {}
    term_cnt = {}
    for ep in endpoints:
        if ep.get("kind") != "terminal":
            continue
        cid = ep.get("comp_id", None)
        if cid is None:
            continue
        term_cnt[cid] = term_cnt.get(cid, 0) + 1

    for comp in components:
        cid = comp.get("id")
        if cid is None:
            continue
        if comp.get("cls_name") != "voltage.dc":
            continue

        n = term_cnt.get(cid, 0)
        if n <= 1:
            patch[cid] = {
                "variant": "one_port",
                "cls_name_refined": "voltage.dc.one_port",
                "matched_terminal_count": int(n),
            }
        else:
            patch[cid] = {
                "variant": "two_port",
                "cls_name_refined": "voltage.dc",
                "matched_terminal_count": int(n),
            }
    return patch


# =======================
# 业务：BJT/FET subtype（用 crops，原逻辑保留）
# =======================
def refine_transistor_subtype(base_name: str, components: List[Dict],
                             bjt_model, bjt_classes, bjt_tf,
                             fet_model, fet_classes, fet_tf) -> Dict[int, Dict]:
    patch = {}

    bjt_comp_ids = [c["id"] for c in components if c.get("cls_name") == "transistor.bjt"]
    fet_comp_ids = [c["id"] for c in components if c.get("cls_name") == "transistor.fet"]

    bjt_crops = list_crops_for_image("transistor.bjt", base_name)
    fet_crops = list_crops_for_image("transistor.fet", base_name)

    n_b = min(len(bjt_comp_ids), len(bjt_crops))
    for i in range(n_b):
        cid = bjt_comp_ids[i]
        crop_path = bjt_crops[i]
        pred, conf = predict_one(bjt_model, bjt_classes, bjt_tf, crop_path)
        patch[cid] = {
            **patch.get(cid, {}),
            "subtype": pred,
            "subtype_conf": conf,
            "subtype_source": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
        }

    n_f = min(len(fet_comp_ids), len(fet_crops))
    for i in range(n_f):
        cid = fet_comp_ids[i]
        crop_path = fet_crops[i]
        pred, conf = predict_one(fet_model, fet_classes, fet_tf, crop_path)
        patch[cid] = {
            **patch.get(cid, {}),
            "subtype": pred,
            "subtype_conf": conf,
            "subtype_source": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
        }

    if len(bjt_comp_ids) != len(bjt_crops):
        patch["_warn_bjt_mismatch"] = {
            "base": base_name,
            "num_components": len(bjt_comp_ids),
            "num_crops": len(bjt_crops),
        }
    if len(fet_comp_ids) != len(fet_crops):
        patch["_warn_fet_mismatch"] = {
            "base": base_name,
            "num_components": len(fet_comp_ids),
            "num_crops": len(fet_crops),
        }

    return patch


# =======================
# ✅ 新增：DC source（二分类 vdc vs idc）
#   关键：用 bbox 匹配 crop，避免 one_port/two_port 混在一起导致顺序错位
# =======================
def parse_bbox_from_crop_name(crop_path: str) -> Optional[Tuple[int, int, int, int]]:
    bn = os.path.basename(crop_path)
    m = re.search(r"_x(-?\d+)y(-?\d+)x(-?\d+)y(-?\d+)_", bn)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return (x1, y1, x2, y2)


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    b_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _center_dist(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def match_crops_by_bbox(
    comp_list: List[Dict],
    crops: List[str],
    *,
    max_center_dist_px: float = 600.0,
) -> Dict[int, str]:
    comp_items: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for comp in comp_list:
        try:
            cid = int(comp.get("id"))
        except Exception:
            continue
        bbox = comp.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        comp_items.append((cid, (x1, y1, x2, y2)))

    crop_items: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    for j, cp in enumerate(crops):
        bb = parse_bbox_from_crop_name(cp)
        if bb is None:
            continue
        crop_items.append((j, cp, bb))

    if not comp_items or not crop_items:
        return {}

    pairs = []
    for cid, cb in comp_items:
        for j, cp, bb in crop_items:
            dist = _center_dist(cb, bb)
            iou = _bbox_iou(cb, bb)
            pairs.append((iou, -dist, cid, j, cp, dist))

    pairs.sort(reverse=True)

    used_crops = set()
    assigned: Dict[int, str] = {}
    for iou, negdist, cid, j, cp, dist in pairs:
        if cid in assigned:
            continue
        if j in used_crops:
            continue
        if dist > max_center_dist_px and iou < 1e-6:
            continue
        assigned[cid] = cp
        used_crops.add(j)

    return assigned


def refine_dc_source_type(
    base_name: str,
    components: List[Dict],
    vdc_variant_patch: Dict[int, Dict],
    dc_model, dc_classes, dc_tf
) -> Dict[int, Dict]:
    """
    只对 two_port 的 voltage.dc 做二分类：
      vdc -> voltage.dc
      idc -> current.dc
    """
    patch: Dict[int, Dict] = {}

    # 取出所有 voltage.dc 的 crops（包含 one_port/two_port 混合）
    dc_crops = list_crops_for_image("voltage.dc", base_name)
    if not dc_crops:
        return patch

    # 只需要对 voltage.dc components 做 bbox 匹配
    dc_comps = [c for c in components if c.get("cls_name") == "voltage.dc"]
    cid2crop = match_crops_by_bbox(dc_comps, dc_crops)

    for comp in dc_comps:
        cid = comp.get("id", None)
        if cid is None:
            continue

        # 必须先经过 one_port/two_port 判定
        vpatch = vdc_variant_patch.get(cid, {})
        refined = vpatch.get("cls_name_refined", "voltage.dc")
        variant = vpatch.get("variant", None)

        # one_port 直接跳过（不需要再分 vdc/idc）
        if refined == "voltage.dc.one_port" or variant == "one_port":
            continue

        crop_path = cid2crop.get(int(cid))
        if not crop_path:
            continue

        pred, conf = predict_one(dc_model, dc_classes, dc_tf, crop_path)

        # 你训练输出 classes=['idc','vdc'] 或 ['vdc','idc'] 都无所谓，按名字判断
        pred_lower = str(pred).lower()

        if pred_lower == "idc":
            patch[int(cid)] = {
                "cls_name_refined": "current.dc",
                "subtype": "idc",
                "subtype_conf": conf,
                "subtype_source": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
                "method": "dinov2_dcsrc",
            }
        elif pred_lower == "vdc":
            patch[int(cid)] = {
                "cls_name_refined": "voltage.dc",
                "subtype": "vdc",
                "subtype_conf": conf,
                "subtype_source": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
                "method": "dinov2_dcsrc",
            }
        else:
            # 兜底：不认识就不改类，只记录一下
            patch[int(cid)] = {
                "note": f"dcsrc_pred_unknown:{pred}",
                "subtype_conf": conf,
                "subtype_source": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
                "method": "dinov2_dcsrc",
            }

    return patch


def main():
    graph_paths = sorted(glob(os.path.join(GRAPH_JSON_DIR, "*_graph.json")))
    if not graph_paths:
        print("[WARN] no graph json found:", GRAPH_JSON_DIR)
        return

    print("[LOAD] Loading DINOv2 classifiers...")
    bjt_model, bjt_classes, bjt_tf = load_dino_classifier(BJT_CKPT)
    fet_model, fet_classes, fet_tf = load_dino_classifier(FET_CKPT)
    dc_model,  dc_classes,  dc_tf  = load_dino_classifier(DCSRC_CKPT)
    print("[LOAD] OK.")
    print("  bjt_classes =", bjt_classes)
    print("  fet_classes =", fet_classes)
    print("  dc_classes  =", dc_classes)

    for gpath in graph_paths:
        gname = os.path.basename(gpath)
        base = gname.replace("_graph.json", "")

        print("\n=== refine:", base, "===")
        image, components, endpoints = load_graph_json(gpath)

        # 1) VDC one-port / two-port
        vdc_patch = refine_vdc_variant(components, endpoints)

        # 2) BJT/FET subtype
        sub_patch = refine_transistor_subtype(
            base, components,
            bjt_model, bjt_classes, bjt_tf,
            fet_model, fet_classes, fet_tf
        )

        # 3) ✅ DC source: VDC vs IDC（二分类）
        dcsrc_patch = refine_dc_source_type(
            base, components, vdc_patch,
            dc_model, dc_classes, dc_tf
        )

        # 合并 patch（按 comp_id）
        comp_patch: Dict[str, Any] = {}

        for cid, d in vdc_patch.items():
            comp_patch[str(cid)] = {**comp_patch.get(str(cid), {}), **d}

        for cid, d in sub_patch.items():
            if isinstance(cid, int):
                comp_patch[str(cid)] = {**comp_patch.get(str(cid), {}), **d}
            else:
                comp_patch[cid] = d

        for cid, d in dcsrc_patch.items():
            comp_patch[str(cid)] = {**comp_patch.get(str(cid), {}), **d}

        out_js = {
            "image": image,
            "base_name": base,
            "patch_by_component_id": comp_patch,
        }

        out_path = os.path.join(OUT_DIR, f"{base}_type_refine.json")
        with open(out_path, "w") as f:
            json.dump(out_js, f, indent=2, ensure_ascii=False)
        print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
