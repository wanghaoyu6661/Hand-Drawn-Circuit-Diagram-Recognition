# -*- coding: utf-8 -*-
"""
infer_ports_vitpose.py  (OFFICIAL TOPDOWN VERSION)

放在 build_connections.py 之后、build_final_json.py 之前运行。

输入：
- link/json/*_graph.json                         (components + endpoints)
- type_refine/json/*_type_refine.json            (cls_name_refined / subtype / variant)
- yolo_detect/exp/crops/<cls_name>/*.jpg         (YOLO 已裁剪的元件图)
  实际使用：/root/autodl-tmp/final_result/yolov10_crops/<base>/<cls_name>/*.jpg

输出：
- /root/autodl-tmp/final_result/ports_cls/json/{base}_ports_patch.json
  只包含“对 endpoints 的端点角色补丁”和“对 components 的端点预测记录”，供 build_final_json.py merge。

说明：
- 改为官方 TopDown 推理：mmpose.apis.init_model + inference_topdown
- 每张 crop 用 bbox=[0,0,w,h]（整图），再用你原来的 map_kps_to_original(...) 映射回原图坐标
- JSON 输出字段保持不变

✅ 本版新增（按你的需求）：
1) 仍然先做“距离最近的端点匹配” + “多余 terminal 合并到目标数量(2/3)”
2) 再做“最高置信度端点种类识别”：
   - 2端点元件：取 matched_eid!=None 的 keypoint 中置信度最高的 1 个作为真；另一个端点补成缺失角色
   - 3端点元件：取 matched_eid!=None 的 keypoint 中置信度最高的 2 个作为真；另一个端点补成缺失角色
3) 输出 kps 时把 (x,y) 吸附到 matched_eid 对应 HAWP terminal 坐标，避免远处错误点污染可视化/匹配
"""

import os
import re
import json
import math
from glob import glob
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image

# ============ OFFICIAL MMPose Topdown APIs ============
from mmpose.apis import init_model, inference_topdown

# ============ 路径 ============
GRAPH_JSON_DIR = "/root/autodl-tmp/final_result/link/json"
TYPE_REFINE_DIR = "/root/autodl-tmp/final_result/type_refine/json"
YOLOV10_CROPS_DIR = "/root/autodl-tmp/final_result/yolov10_crops"

# 你的 ViTPose configs + weights
POSE3K_CFG = "/root/autodl-tmp/mmpose/configs/ports/vitpose_l_ports_3k_256x192_fix.py"
POSE3K_WTS = "/root/autodl-tmp/work_dirs/vitpose_ports_3k_l_256x192/best_coco_AP_epoch_29.pth"

POSE2K_CFG = "/root/autodl-tmp/mmpose/configs/ports/vitpose_l_ports_2k_256x192_fix.py"
POSE2K_WTS = "/root/autodl-tmp/work_dirs/vitpose_ports_2k_l_256x192/best_coco_AP_epoch_30.pth"

OUT_DIR = "/root/autodl-tmp/final_result/ports_cls/json"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda"  # "cpu" 也行

# ============ 匹配阈值（keypoint -> endpoint）===========
MATCH_TH_PX = 30.0  # fallback default

def adaptive_match_th_px(bbox, img_wh):
    """
    bbox: [x1,y1,x2,y2] in original image space
    img_wh: (W,H) of original image
    思路：阈值随元件尺寸 + 图像尺寸轻微放大，并做 clamp，避免过大/过小
    """
    try:
        W, H = img_wh
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
    except Exception:
        return float(MATCH_TH_PX)

    # 元件尺度（对角线）
    diag = (bw * bw + bh * bh) ** 0.5

    # 你可以调的几个比例（建议先用这组）
    th_from_comp = 0.18 * diag           # 端点允许偏离≈元件对角线的18%
    th_from_img  = 0.006 * min(W, H)     # 随图大小的保底项（4000图≈24px）

    th = max(th_from_comp, th_from_img)

    # clamp：避免小元件太严/大图太松
    th = max(12.0, min(th, 80.0))
    return float(th)


# crop params (must match make_yolov10_crops_by_image.py)
CROP_SIZE = 200
OUT_SIZE  = 320


# =======================
# keypoint index -> role name 映射
# =======================
KP_NAMES_3K = {
    "transistor.bjt": ["Base", "Collector", "Emitter"],
    "transistor.fet": ["Gate", "Drain", "Source"],
    "operational_amplifier": ["In+", "In-", "Out"],
    "operational_amplifier.schmitt_trigger": ["In+", "In-", "Out"],
}

KP_NAMES_2K = {
    "diode": ["Anode", "Cathode"],
    "diode.zener": ["Anode", "Cathode"],
    "voltage.dc": ["Positive", "Negative"],
    "current.dc": ["Flowing From", "Flowing To"],
}


# =======================
# utils
# =======================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

SRC_IMG_DIR = "/root/autodl-tmp/final_result/src_img"

def get_image_wh(base, image_name):
    """
    尽量从 image_name 找到原图；找不到就去 SRC_IMG_DIR 里按 base 猜扩展名。
    """
    cand = []
    if image_name:
        cand.append(image_name)
        cand.append(os.path.join(SRC_IMG_DIR, image_name))
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        cand.append(os.path.join(SRC_IMG_DIR, base + ext))

    for p in cand:
        try:
            if p and os.path.exists(p):
                w, h = Image.open(p).size
                return int(w), int(h)
        except Exception:
            pass
    return None  # 允许失败

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


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_graph(gpath: str) -> Tuple[str, List[Dict], List[Dict]]:
    js = load_json(gpath)
    return js.get("image", ""), js.get("components", []), js.get("endpoints", [])


def load_type_refine(base: str) -> Dict[str, Any]:
    path = os.path.join(TYPE_REFINE_DIR, f"{base}_type_refine.json")
    if not os.path.exists(path):
        return {}
    js = load_json(path)
    return js.get("patch_by_component_id", {})


def get_refined_cls_and_subtype(type_patch: Dict[str, Any], comp_id: int, fallback_cls: str) -> Tuple[str, Optional[str]]:
    d = type_patch.get(str(comp_id), {})
    cls_refined = d.get("cls_name_refined", fallback_cls)
    subtype = d.get("subtype", None)
    return cls_refined, subtype


def parse_bbox_from_crop_name(crop_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse original-image bbox (pixel coords) from crop filename produced by make_yolov10_crops_by_image.py.
    Example:
      0003_x736y55x793y172_c0.87.jpg
    """
    bn = os.path.basename(crop_path)
    m = re.search(r"_x(-?\d+)y(-?\d+)x(-?\d+)y(-?\d+)_", bn)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return (x1, y1, x2, y2)


def parse_crop_size_from_crop_name(crop_path: str) -> Optional[int]:
    """
    新 crop 文件名含 _s{crop_size}_c
    例如：..._s320_c0.87.jpg
    """
    bn = os.path.basename(crop_path)
    m = re.search(r"_s(\d+)_c", bn)
    if not m:
        return None
    return int(m.group(1))


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
    """
    Robustly match crop images to components using geometry instead of list order.
    Returns: {comp_id(int): crop_path(str)}
    """
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


def map_kps_to_original_center_crop(kps_xy: List[Tuple[float, float]], crop_path: str) -> Optional[List[List[float]]]:
    bbox = parse_bbox_from_crop_name(crop_path)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    crop_size = parse_crop_size_from_crop_name(crop_path) or CROP_SIZE
    half = crop_size / 2.0

    tx1 = int(round(cx - half))
    ty1 = int(round(cy - half))

    s_inv = float(crop_size) / float(OUT_SIZE)

    out = []
    for (rx, ry) in kps_xy:
        x_canvas = float(rx) * s_inv
        y_canvas = float(ry) * s_inv
        out.append([float(tx1 + x_canvas), float(ty1 + y_canvas)])
    return out


def map_kps_to_original(crop_path: str, bbox_xyxy: List[float], kps_xy: List[List[float]]) -> List[List[float]]:
    mapped = map_kps_to_original_center_crop([(float(x), float(y)) for x, y in kps_xy], crop_path)
    if mapped is not None:
        return mapped

    # fallback
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)

    w, h = Image.open(crop_path).size
    sx = bw / max(1e-6, float(w))
    sy = bh / max(1e-6, float(h))

    out = []
    for (cx, cy) in kps_xy:
        out.append([float(x1 + cx * sx), float(y1 + cy * sy)])
    return out


def greedy_match_roles_to_endpoints(
    roles_xy: List[Tuple[str, float, float, float]],  # (role, x, y, conf)
    terminals: List[Dict],
    th: float = MATCH_TH_PX
) -> List[Dict]:
    if not terminals:
        return [{"role": r, "x": x, "y": y, "conf": c, "matched_eid": None, "dist": None}
                for (r, x, y, c) in roles_xy]

    unused = set(ep["eid"] for ep in terminals)
    ep_by_eid = {ep["eid"]: ep for ep in terminals}

    results = []
    for role, x, y, conf in roles_xy:
        best = (1e18, None)
        for eid in list(unused):
            ep = ep_by_eid[eid]
            d = math.hypot(ep["x"] - x, ep["y"] - y)
            if d < best[0]:
                best = (d, eid)

        dist, eid = best
        if eid is not None and dist <= th:
            unused.remove(eid)
            results.append({"role": role, "x": x, "y": y, "conf": conf,
                            "matched_eid": int(eid), "dist": float(dist)})
        else:
            results.append({"role": role, "x": x, "y": y, "conf": conf,
                            "matched_eid": None, "dist": None})
    return results

def adaptive_max_center_dist_px(img_wh):
    """
    crop bbox 与 comp bbox 的中心距离上限（只在 iou≈0 时起作用）
    """
    try:
        W, H = img_wh
    except Exception:
        return 600.0

    # 比例项：你可以调这个系数
    v = 0.15 * min(W, H)   # 4000图≈600px，1000图≈150px
    v = max(180.0, min(v, 900.0))  # clamp
    return float(v)

def refine_roles_by_confidence(
    kp_roles: List[str],
    matched: List[Dict],
    terminals: List[Dict],
) -> List[Dict]:
    """
    matched: greedy_match_roles_to_endpoints 的输出 list[{"role","conf","matched_eid",...}]
    terminals: 当前 comp 的 terminals（已经过“合并到目标数量”）
    目标：
      - 2K: 取 conf 最大的 1 个 role 作为可信；另一个 eid 直接补成另一角色
      - 3K: 取 conf 最大的 2 个 role 作为可信；剩下的 eid 补成剩余角色
    注意：
      - 只有当 terminals 数量 == len(kp_roles) 才做（即 2 或 3）
      - 必须至少有 1(2K)/2(3K) 个 matched_eid 非 None 才做；否则不动
      - 如果出现“两个 role 命中同一个 eid”，按 conf 取高者，低者当作未命中
    返回：新的 matched（会把补全的 role 也写成 matched_eid != None）
    """
    K = len(kp_roles)
    if len(terminals) != K:
        return matched

    term_eids = [int(t["eid"]) for t in terminals]
    # 只保留 matched_eid 合法的项
    valid = []
    for it in matched:
        me = it.get("matched_eid", None)
        if me is None:
            continue
        me = int(me)
        if me in term_eids:
            valid.append({**it, "matched_eid": me})

    if K == 2:
        if len(valid) < 1:
            return matched
        # 选 conf 最大的一个
        best = max(valid, key=lambda x: float(x.get("conf", 0.0)))
        best_eid = int(best["matched_eid"])
        best_role = best["role"]

        # 找另一个 eid 和另一个 role
        other_eid = [e for e in term_eids if e != best_eid]
        if not other_eid:
            return matched
        other_eid = int(other_eid[0])

        other_role = [r for r in kp_roles if r != best_role]
        if not other_role:
            return matched
        other_role = other_role[0]

        # 重新组织输出：两个端点都保证有 matched_eid
        out = []
        out.append({**best, "matched_eid": best_eid, "dist": best.get("dist", None)})
        out.append({"role": other_role, "x": None, "y": None, "conf": 0.0, "matched_eid": other_eid, "dist": None})
        return out

    if K == 3:
        if len(valid) < 2:
            return matched

        # 去重：同一个 eid 只留 conf 最大的一条
        best_by_eid = {}
        for it in valid:
            eid = int(it["matched_eid"])
            c = float(it.get("conf", 0.0))
            if (eid not in best_by_eid) or (c > float(best_by_eid[eid].get("conf", 0.0))):
                best_by_eid[eid] = it
        uniq = list(best_by_eid.values())
        if len(uniq) < 2:
            return matched

        # 取 conf 最大的 2 条
        uniq.sort(key=lambda x: float(x.get("conf", 0.0)), reverse=True)
        top2 = uniq[:2]
        used_eids = {int(x["matched_eid"]) for x in top2}
        used_roles = {x["role"] for x in top2}

        # 找剩余 eid / role
        remain_eids = [e for e in term_eids if e not in used_eids]
        remain_roles = [r for r in kp_roles if r not in used_roles]
        if len(remain_eids) != 1 or len(remain_roles) != 1:
            return matched

        out = []
        out.extend(top2)
        out.append({"role": remain_roles[0], "x": None, "y": None, "conf": 0.0,
                    "matched_eid": int(remain_eids[0]), "dist": None})
        return out

    return matched

# =======================
# ✅ 新增：基于“已匹配到 HAWP terminal 的 keypoint”做最高置信度角色决策
# 目标：过滤掉少量“跑偏很远”的错误 keypoint（通常 matched_eid=None），
# 只用最可信的 1(2K) / 2(3K) 个端点角色，然后把剩余 terminal 直接补成缺失的角色。
#
# 注意：
# - 这一步应当在 _merge_terminals_to_target_k() 把 terminals 合并到目标数量后再做
# - 只使用 matched_eid!=None 的 keypoint 参与“最高置信度”选择
# - 输出时把 kps 的 (x,y) 统一“吸附”到 matched_eid 对应的 HAWP terminal 坐标，避免可视化出现远离元件的点
# =======================
def _snap_kps_xy_to_terminals(kps: List[Dict], terminals: List[Dict]) -> List[Dict]:
    if not kps or not terminals:
        return kps
    by = {int(t["eid"]): t for t in terminals if "eid" in t}
    out = []
    for it in kps:
        it2 = dict(it)
        me = it2.get("matched_eid", None)
        if me is not None and int(me) in by:
            t = by[int(me)]
            it2["x"] = float(t["x"])
            it2["y"] = float(t["y"])
        out.append(it2)
    return out


def _finalize_roles_by_best_conf(
    kps_matched: List[Dict],
    terminals: List[Dict],
    kp_roles: List[str],
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    输入：
      - kps_matched: greedy_match_roles_to_endpoints 的输出（每个 role 对应一个 kp 记录）
      - terminals: 当前 comp 的 terminal 列表（已合并到目标数量：len(terminals)=K）
      - kp_roles: 该元件的角色列表（长度=K）

    输出：
      - kps_final: 只包含 K 条记录（每个 role 一条），且每条都有 matched_eid（若 terminals 足够）
      - role_patch: {eid(str): {"port_role": role, "port_conf": conf, "port_from": ...}}（只含本 comp）
    """
    K = len(kp_roles)
    if K <= 0:
        return kps_matched, {}

    # terminals by eid
    t_by = {int(t["eid"]): t for t in terminals if "eid" in t}

    # 只用“确实匹配到 terminal”的 kp 参与“最高置信度”选择（过滤跑偏点）
    candidates = []
    for it in kps_matched or []:
        me = it.get("matched_eid", None)
        role = it.get("role", None)
        conf = it.get("conf", None)
        if me is None or role is None:
            continue
        if int(me) not in t_by:
            continue
        try:
            c = float(conf) if conf is not None else 0.0
        except Exception:
            c = 0.0
        candidates.append((c, str(role), int(me), it))

    # 按置信度从高到低
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 选择最可信的 K-1 个“不同角色”的端点（2K -> 1个，3K -> 2个）
    need = max(0, K - 1)
    picked_roles = set()
    picked_eids = set()
    picked = []  # (conf, role, eid, raw_it)
    for c, role, eid, raw in candidates:
        if len(picked) >= need:
            break
        if role in picked_roles:
            continue
        if eid in picked_eids:
            continue
        picked.append((c, role, eid, raw))
        picked_roles.add(role)
        picked_eids.add(eid)

    # 退化：如果一个都没选到 -> 用 dist 最小的那个当种子
    if need > 0 and len(picked) == 0:
        tmp = []
        for it in kps_matched or []:
            me = it.get("matched_eid", None)
            role = it.get("role", None)
            if me is None or role is None:
                continue
            if int(me) not in t_by:
                continue
            d = it.get("dist", None)
            try:
                dd = float(d) if d is not None else 1e18
            except Exception:
                dd = 1e18
            conf = it.get("conf", None)
            try:
                c = float(conf) if conf is not None else 0.0
            except Exception:
                c = 0.0
            tmp.append((dd, -c, str(role), int(me), it))
        tmp.sort()
        if tmp:
            dd, negc, role, eid, raw = tmp[0]
            picked.append((-negc, role, eid, raw))
            picked_roles.add(role)
            picked_eids.add(eid)

    # 缺失角色 & 缺失 eid
    missing_roles = [r for r in kp_roles if r not in picked_roles]
    missing_eids = [int(t["eid"]) for t in terminals if int(t["eid"]) not in picked_eids]

    inferred = []
    for r, eid in zip(missing_roles, missing_eids):
        inferred.append((r, eid))

    # 生成最终 kps（每个 role 一条），并“吸附”到 terminal 坐标
    best_conf = picked[0][0] if picked else 0.0
    second_conf = picked[1][0] if len(picked) > 1 else best_conf

    role_to_entry = {}

    for (c, role, eid, raw) in picked:
        t = t_by.get(int(eid))
        role_to_entry[role] = {
            "role": role,
            "x": float(t["x"]) if t is not None else float(raw.get("x", 0.0)),
            "y": float(t["y"]) if t is not None else float(raw.get("y", 0.0)),
            "conf": float(c),
            "matched_eid": int(eid),
            "dist": float(raw.get("dist")) if raw.get("dist") is not None else None,
            "inferred": False,
        }

    for j, (role, eid) in enumerate(inferred):
        t = t_by.get(int(eid))
        # 2K: 用 best_conf；3K: 用 best_conf/second_conf 的最小值更保守
        c = best_conf if K == 2 else (min(best_conf, second_conf) if need == 2 else best_conf)
        role_to_entry[role] = {
            "role": role,
            "x": float(t["x"]) if t is not None else 0.0,
            "y": float(t["y"]) if t is not None else 0.0,
            "conf": float(c),
            "matched_eid": int(eid),
            "dist": 0.0,
            "inferred": True,
        }

    kps_final = []
    for role in kp_roles:
        if role in role_to_entry:
            kps_final.append(role_to_entry[role])
        else:
            eid = None
            for t in terminals:
                te = int(t["eid"])
                if te not in picked_eids:
                    eid = te
                    break
            if eid is None and terminals:
                eid = int(terminals[0]["eid"])
            t = t_by.get(int(eid)) if eid is not None else None
            kps_final.append({
                "role": role,
                "x": float(t["x"]) if t is not None else 0.0,
                "y": float(t["y"]) if t is not None else 0.0,
                "conf": float(best_conf),
                "matched_eid": int(eid) if eid is not None else None,
                "dist": 0.0,
                "inferred": True,
            })

    role_patch = {}
    for it in kps_final:
        me = it.get("matched_eid", None)
        if me is None:
            continue
        role_patch[str(int(me))] = {
            "port_role": it.get("role", None),
            "port_conf": it.get("conf", None),
            "port_from": "vitpose_bestconf" if not it.get("inferred", False) else "vitpose_bestconf_infer",
        }

    return kps_final, role_patch


# =======================
# ✅ 新增：当某些 2/3 端点元件被 build_connections.py 误检测出“多余 terminal”时，
# 在做 ViTPose 角色匹配前，先把 terminals 按“最近距离”迭代合并到目标数量。
# 合并规则：
#   - 每次找距离最近的一对 terminal
#   - 取中心点 (midpoint) 作为新坐标
#   - 保留 eid 较小的那个作为代表点（坐标更新为中心点）
#   - eid 较大的那个记为 merged（后续在 build_final_json.py 里把边引用替换为代表点并删除该点）
# 输出到 ports_patch.json：
#   endpoint_merge_patch: { dropped_eid: {"merge_into": keep_eid} }
#   endpoint_coord_patch: { keep_eid: {"x": new_x, "y": new_y} }
# =======================
def _merge_terminals_to_target_k(terminals: List[Dict], target_k: int):
    """
    terminals: graph_json 里的 terminal endpoints（含 eid,x,y,comp_id...）
    返回:
      merged_terms: List[Dict]   (只含保留下来的 eid)
      merge_patch: Dict[str, Dict]   dropped_eid -> {"merge_into": keep_eid}
      coord_patch: Dict[str, Dict]   keep_eid -> {"x": new_x, "y": new_y}
    """
    if not terminals or len(terminals) <= target_k:
        return terminals, {}, {}

    # --- DSU / Union-Find ---
    parent: Dict[int, int] = {}
    def find(a: int) -> int:
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        keep = min(ra, rb)
        drop = max(ra, rb)
        parent[drop] = keep
        return keep

    rep_xy: Dict[int, Tuple[float, float]] = {}
    for ep in terminals:
        eid = int(ep["eid"])
        x, y = float(ep["x"]), float(ep["y"])
        parent[eid] = eid
        rep_xy[eid] = (x, y)

    reps = set(parent.keys())

    def dist2(ea: int, eb: int) -> float:
        xa, ya = rep_xy[ea]
        xb, yb = rep_xy[eb]
        dx, dy = xa - xb, ya - yb
        return dx*dx + dy*dy

    while len(reps) > target_k:
        reps_list = sorted(reps)
        best = (1e30, None, None)
        for i in range(len(reps_list)):
            for j in range(i + 1, len(reps_list)):
                a, b = reps_list[i], reps_list[j]
                d2 = dist2(a, b)
                if d2 < best[0]:
                    best = (d2, a, b)

        _, a, b = best
        if a is None or b is None:
            break

        xa, ya = rep_xy[a]
        xb, yb = rep_xy[b]
        mx, my = (xa + xb) * 0.5, (ya + yb) * 0.5

        keep = union(a, b)
        drop = a if keep == b else b

        rep_xy[keep] = (mx, my)

        if drop in reps:
            reps.remove(drop)
        reps.add(keep)

    merge_patch: Dict[str, Dict] = {}
    for ep in terminals:
        eid = int(ep["eid"])
        rep = find(eid)
        if rep != eid:
            merge_patch[str(eid)] = {"merge_into": int(rep)}

    coord_patch: Dict[str, Dict] = {}
    for rep in set(find(int(ep["eid"])) for ep in terminals):
        if rep in rep_xy:
            coord_patch[str(rep)] = {"x": float(rep_xy[rep][0]), "y": float(rep_xy[rep][1])}

    merged_terms = []
    keep_reps = set(find(int(ep["eid"])) for ep in terminals)
    for ep in terminals:
        eid = int(ep["eid"])
        if find(eid) == eid and eid in keep_reps:
            ep2 = dict(ep)
            if str(eid) in coord_patch:
                ep2["x"] = float(coord_patch[str(eid)]["x"])
                ep2["y"] = float(coord_patch[str(eid)]["y"])
            merged_terms.append(ep2)

    merged_terms = sorted(merged_terms, key=lambda x: int(x["eid"]))
    return merged_terms, merge_patch, coord_patch


def run_official_topdown(model, img_path: str) -> Tuple[List[List[float]], List[float]]:
    """
    兼容版 official topdown：
    - 优先走新版：inference_topdown(model, img_path, bboxes=(N,4))
    - 失败则 fallback 旧版：inference_topdown(model, img, [person], bbox_format="xyxy")
    """
    # 读尺寸
    w, h = Image.open(img_path).size

    # ✅ 强制 (1,4) xyxy
    bboxes = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)

    # -------- 1) 新版 API 路径（你当前文件里就是这种写法）--------
    try:
        results = inference_topdown(model, img_path, bboxes=bboxes)
        if results:
            ds = results[0]
            pi = getattr(ds, "pred_instances", None)
            if pi is None:
                return [], []
            kps = np.array(getattr(pi, "keypoints", None), dtype=np.float32)
            scs = getattr(pi, "keypoint_scores", None)

            if kps.size == 0:
                return [], []
            if kps.ndim == 3:  # (1,K,2)
                kps = kps[0]
            kps_xy = [[float(x), float(y)] for x, y in kps.tolist()]

            if scs is None:
                kps_score = [1.0] * len(kps_xy)
            else:
                scs = np.array(scs, dtype=np.float32)
                if scs.ndim == 2:
                    scs = scs[0]
                kps_score = [float(s) for s in scs.tolist()]
            return kps_xy, kps_score
    except Exception:
        pass

    # -------- 2) 旧版 API fallback（避免 KeyError: None / bbox 结构问题）--------
    img = np.array(Image.open(img_path).convert("RGB"))
    hh, ww = img.shape[:2]
    person = {"bbox": np.array([0.0, 0.0, float(ww), float(hh)], dtype=np.float32)}  # ✅ 4维

    results = inference_topdown(model, img, [person], bbox_format="xyxy")
    if not results:
        return [], []

    # 旧版返回 dict 的情况
    r0 = results[0]
    inst = r0.get("pred_instances", None) if isinstance(r0, dict) else getattr(r0, "pred_instances", None)
    if inst is None:
        return [], []

    kps = np.array(inst.keypoints[0], dtype=np.float32)       # (K,2)
    scs = np.array(inst.keypoint_scores[0], dtype=np.float32) # (K,)
    kps_xy = [[float(x), float(y)] for x, y in kps.tolist()]
    kps_sc = [float(x) for x in scs.tolist()]
    return kps_xy, kps_sc




def main():
    json_list = sorted(glob(os.path.join(GRAPH_JSON_DIR, "*_graph.json")))
    if not json_list:
        print("[WARN] no graph json found in", GRAPH_JSON_DIR)
        return

    print("[LOAD] init_model (official topdown) ...")
    model3k = init_model(POSE3K_CFG, POSE3K_WTS, device=DEVICE)
    model2k = init_model(POSE2K_CFG, POSE2K_WTS, device=DEVICE)
    print("[LOAD] OK")

    for gpath in json_list:
        base = os.path.splitext(os.path.basename(gpath))[0].replace("_graph", "")
        print(f"\n=== ports infer: {base} ===")

        image_name, components, endpoints = load_graph(gpath)
        
        img_wh = get_image_wh(base, image_name) or (2000, 2000)  # 实在取不到就给个兜底

        # type refine patch（可能为空）
        type_patch = load_type_refine(base)

        # terminals_by_comp
        terminals_by_comp: Dict[int, List[Dict]] = {}
        for ep in endpoints:
            if ep.get("kind") != "terminal":
                continue
            cid = ep.get("comp_id", None)
            if cid is None:
                continue
            terminals_by_comp.setdefault(int(cid), []).append(ep)

        # 收集 3K/2K 目标元件（按 YOLO 原始类分桶，crop 文件夹用原始类）
        comps_3k = {k: [] for k in KP_NAMES_3K.keys()}
        comps_2k = {k: [] for k in KP_NAMES_2K.keys()}

        for comp in components:
            cid = comp.get("id", None)
            cls0 = comp.get("cls_name", None)
            if cid is None or cls0 is None:
                continue

            # 细分类结果（决定端点语义）
            cls_refined, subtype = get_refined_cls_and_subtype(type_patch, int(cid), cls0)

            # voltage.dc.one_port 不跑端点
            if cls_refined == "voltage.dc.one_port":
                continue

            if cls0 in comps_3k:
                comp2 = dict(comp)
                comp2["_cls_refined"] = cls_refined
                comp2["_subtype"] = subtype
                comps_3k[cls0].append(comp2)

            if cls0 in comps_2k:
                comp2 = dict(comp)
                comp2["_cls_refined"] = cls_refined
                comp2["_subtype"] = subtype
                comps_2k[cls0].append(comp2)

        endpoint_role_patch: Dict[str, Dict] = {}
        component_ports: Dict[str, Any] = {}

        endpoint_merge_patch: Dict[str, Dict] = {}
        endpoint_coord_patch: Dict[str, Dict] = {}

        # --------- 处理 3K 类 ---------
        for cls_name, comp_list in comps_3k.items():
            if not comp_list:
                continue
            crops = list_crops_for_image(cls_name, base)
            maxd = adaptive_max_center_dist_px(img_wh)
            mapping = match_crops_by_bbox(comp_list, crops, max_center_dist_px=maxd)
            if len(mapping) != len(comp_list):
                print(f"[WARN] 3K bbox-match {cls_name}: comps={len(comp_list)} crops={len(crops)} matched={len(mapping)}")

            kp_roles = KP_NAMES_3K[cls_name]

            for comp in comp_list:
                cid = int(comp["id"])
                bbox = comp.get("bbox", None)
                if not bbox:
                    continue

                crop_path = mapping.get(cid, None)
                if crop_path is None:
                    print(f"[WARN] 3K no crop matched by bbox for {cls_name} comp_id={cid}")
                    continue

                kps_xy, kps_sc = run_official_topdown(model3k, crop_path)
                if len(kps_xy) != 3:
                    print(f"[WARN] 3K bad kps len={len(kps_xy)} for {crop_path}")
                    continue

                kps_org = map_kps_to_original(crop_path, bbox, kps_xy)
                if not kps_org:
                    print(f"[WARN] 3K map back failed: {crop_path}")
                    continue

                roles_xy = []
                for ridx, (xy, sc) in enumerate(zip(kps_org, kps_sc)):
                    role = kp_roles[ridx]
                    roles_xy.append((role, float(xy[0]), float(xy[1]), float(sc)))

                terms = terminals_by_comp.get(cid, [])

                # ✅ 处理多余 terminal：固定 3 端点器件才合并；op-AMP 允许多端点，不合并
                OPAMP_MULTI = {"operational_amplifier", "operational_amplifier.schmitt_trigger"}

                if (cls_name not in OPAMP_MULTI) and (len(terms) > 3):
                    terms2, mp, cp = _merge_terminals_to_target_k(terms, 3)
                    if mp:
                        endpoint_merge_patch.update(mp)
                    if cp:
                        endpoint_coord_patch.update(cp)
                    terms = terms2

                th = adaptive_match_th_px(bbox, img_wh)
                matched = greedy_match_roles_to_endpoints(roles_xy, terms, th=th)
                matched = _snap_kps_xy_to_terminals(matched, terms)

                is_opamp_multi = (cls_name in OPAMP_MULTI) and (len(terms) > 3)

                if not is_opamp_multi:
                    matched = refine_roles_by_confidence(kp_roles, matched, terms)
                    kps_final, role_patch_local = _finalize_roles_by_best_conf(matched, terms, kp_roles)
                else:
                    # op-AMP 多端点：只信 greedy 的 3 个角色匹配，不做“补全到某个剩余 eid”
                    kps_final = matched
                    role_patch_local = {}
                    for it in matched:
                        me = it.get("matched_eid", None)
                        if me is None:
                            continue
                        role_patch_local[str(int(me))] = {
                            "port_role": it.get("role", None),
                            "port_conf": it.get("conf", None),
                            "port_from": "vitpose_greedy_opamp_multi",
                        }

                component_ports[str(cid)] = {
                    "cls_name": cls_name,
                    "cls_name_refined": comp.get("_cls_refined", cls_name),
                    "subtype": comp.get("_subtype", None),
                    "bbox": bbox,
                    "crop_rel": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
                    "kps": kps_final,
                    "kps_raw": matched,
                }

                for eid_str, patch in role_patch_local.items():
                    endpoint_role_patch[eid_str] = {
                        **patch,
                        "comp_id": cid,
                    }

        # --------- 处理 2K 类 ---------
        for cls_name, comp_list in comps_2k.items():
            if not comp_list:
                continue
            crops = list_crops_for_image(cls_name, base)
            maxd = adaptive_max_center_dist_px(img_wh)
            mapping = match_crops_by_bbox(comp_list, crops, max_center_dist_px=maxd)
            if len(mapping) != len(comp_list):
                print(f"[WARN] 2K bbox-match {cls_name}: comps={len(comp_list)} crops={len(crops)} matched={len(mapping)}")

            for comp in comp_list:
                cid = int(comp["id"])
                bbox = comp.get("bbox", None)
                if not bbox:
                    continue

                crop_path = mapping.get(cid, None)
                if crop_path is None:
                    print(f"[WARN] 2K no crop matched by bbox for {cls_name} comp_id={cid}")
                    continue

                kps_xy, kps_sc = run_official_topdown(model2k, crop_path)
                if len(kps_xy) != 2:
                    print(f"[WARN] 2K bad kps len={len(kps_xy)} for {crop_path}")
                    continue

                kps_org = map_kps_to_original(crop_path, bbox, kps_xy)
                if not kps_org:
                    print(f"[WARN] 2K map back failed: {crop_path}")
                    continue

                # ✅ 按 refined 类名选择角色表（voltage.dc 可能被 refine 成 current.dc）
                kp_key = comp.get("_cls_refined", cls_name)
                if kp_key not in KP_NAMES_2K:
                    kp_key = cls_name
                kp_roles = KP_NAMES_2K[kp_key]

                roles_xy = []
                for ridx, (xy, sc) in enumerate(zip(kps_org, kps_sc)):
                    role = kp_roles[ridx]
                    roles_xy.append((role, float(xy[0]), float(xy[1]), float(sc)))

                terms = terminals_by_comp.get(cid, [])

                # ✅ 处理多余 terminal：先合并到 2 个
                if len(terms) > 2:
                    terms2, mp, cp = _merge_terminals_to_target_k(terms, 2)
                    if mp:
                        endpoint_merge_patch.update(mp)
                    if cp:
                        endpoint_coord_patch.update(cp)
                    terms = terms2

                th = adaptive_match_th_px(bbox, img_wh)
                matched = greedy_match_roles_to_endpoints(roles_xy, terms, th=th)
                matched = refine_roles_by_confidence(kp_roles, matched, terms)  # ✅ 新增：最高置信度补全


                # ✅ 吸附到 HAWP terminal 坐标，避免“跑偏点”影响后续决策/可视化
                matched = _snap_kps_xy_to_terminals(matched, terms)

                # ✅ 最高置信度决策（2K：取最可信1个，其余1个补全）
                kps_final, role_patch_local = _finalize_roles_by_best_conf(matched, terms, kp_roles)

                component_ports[str(cid)] = {
                    "cls_name": cls_name,
                    "cls_name_refined": comp.get("_cls_refined", cls_name),
                    "subtype": comp.get("_subtype", None),
                    "bbox": bbox,
                    "crop_rel": os.path.relpath(crop_path, YOLOV10_CROPS_DIR),
                    "kps": kps_final,
                    "kps_raw": matched,
                }

                for eid_str, patch in role_patch_local.items():
                    endpoint_role_patch[eid_str] = {
                        **patch,
                        "comp_id": cid,
                    }

        out_js = {
            "image": image_name,
            "base_name": base,
            "endpoint_role_patch": endpoint_role_patch,
            "endpoint_merge_patch": endpoint_merge_patch,
            "endpoint_coord_patch": endpoint_coord_patch,
            "component_ports": component_ports,
        }

        out_path = os.path.join(OUT_DIR, f"{base}_ports_patch.json")
        with open(out_path, "w") as f:
            json.dump(out_js, f, indent=2, ensure_ascii=False)
        print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
