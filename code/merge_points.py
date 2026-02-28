import os
import cv2
import json
from glob import glob

import numpy as np
from sklearn.cluster import DBSCAN

# =========================
# 输入 / 输出目录
# =========================
HAWP_JSON_DIR = "/root/autodl-tmp/final_result/HAWPimg/json"
YOLO_LABEL_DIR = "/root/autodl-tmp/final_result/yolo_detect/exp/labels"
IMG_DIR = "/root/autodl-tmp/final_result/lama_clean"

OUT_VIS = "/root/autodl-tmp/final_result/merged_points/merged_points_vis"
OUT_JSON = "/root/autodl-tmp/final_result/merged_points/merged_points_json"

os.makedirs(OUT_VIS, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

# =========================
# YOLO 类别名（来自 data.yaml）
# =========================
YOLO_NAMES = [
    "text",
    "junction",
    "crossover",
    "terminal",
    "gnd",
    "vss",
    "voltage.dc",
    "voltage.ac",
    "voltage.battery",
    "resistor",
    "resistor.adjustable",
    "resistor.photo",
    "capacitor.unpolarized",
    "capacitor.polarized",
    "capacitor.adjustable",
    "inductor",
    "inductor.ferrite",
    "inductor.coupled",
    "transformer",
    "diode",
    "diode.light_emitting",
    "diode.thyrector",
    "diode.zener",
    "diac",
    "triac",
    "thyristor",
    "varistor",
    "transistor.bjt",
    "transistor.fet",
    "transistor.photo",
    "operational_amplifier",
    "operational_amplifier.schmitt_trigger",
    "optocoupler",
    "integrated_circuit",
    "integrated_circuit.ne555",
    "integrated_circuit.voltage_regulator",
    "xor",
    "and",
    "or",
    "not",
    "nand",
    "nor",
    "probe",
    "probe.current",
    "probe.voltage",
    "switch",
    "relay",
    "socket",
    "fuse",
    "speaker",
    "motor",
    "lamp",
    "microphone",
    "antenna",
    "crystal",
    "mechanical",
    "magnetic",
    "optical",
]

# 这些类别不参与“点-元件”端点匹配
SKIP_COMP_CLASSES = {0, 1}  # 0:text, 1:junction

# 与 clean_with_yolov9_lama.py 保持一致的扩展像素
EXPAND_PIXELS_INTEGRATED_CIRCUIT = 0   # 对类 33/34/35
EXPAND_PIXELS_OTHER = 0                 # 对其它所有类
IC_CLASSES = {33, 34, 35}

# 匹配点到 bbox 边的阈值（像素）
EDGE_DIST_TH = 8.0   # 点到边的垂直距离阈值
EDGE_EXTEND = 5.0    # 边在切向方向上的放宽


# =========================
# 基础加载函数
# =========================
def load_hawp_points(json_path):
    """从 HAWP 输出 JSON 中读取 junctions_pred"""
    with open(json_path, "r") as f:
        js = json.load(f)
    pts = js.get("junctions_pred", [])
    return [(float(x), float(y)) for x, y in pts]


def load_yolo_junction_points(txt_path, img_w, img_h, junc_class=1):
    """从 YOLO txt 中读取 junction 类（class=1）的点，转换到像素坐标"""
    pts = []
    if not os.path.exists(txt_path):
        return pts

    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls_id = int(p[0])
            if cls_id != junc_class:
                continue
            xc, yc = map(float, p[1:3])
            pts.append((xc * img_w, yc * img_h))
    return pts


def load_yolo_components(txt_path, img_w, img_h):
    """
    从 YOLO txt 中读取所有“元件框”（除 text/junction），
    并按照 clean_with_yolov9_lama.py 的逻辑对 bbox 进行扩展。
    """
    components = []
    if not os.path.exists(txt_path):
        return components

    comp_id = 0
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls_id = int(p[0])
            if cls_id in SKIP_COMP_CLASSES:
                continue

            xc, yc, w, h = map(float, p[1:5])
            bw = w * img_w
            bh = h * img_h
            cx = xc * img_w
            cy = yc * img_h
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            # 与 clean_with_yolov9_lama.py 保持一致的扩展逻辑
            if cls_id in IC_CLASSES:
                expand = EXPAND_PIXELS_INTEGRATED_CIRCUIT
            else:
                expand = EXPAND_PIXELS_OTHER

            x1 -= expand
            y1 -= expand
            x2 += expand
            y2 += expand

            # 裁剪到图像范围内
            x1 = max(0.0, min(x1, img_w - 1.0))
            y1 = max(0.0, min(y1, img_h - 1.0))
            x2 = max(0.0, min(x2, img_w - 1.0))
            y2 = max(0.0, min(y2, img_h - 1.0))

            cls_name = YOLO_NAMES[cls_id] if 0 <= cls_id < len(YOLO_NAMES) else f"class_{cls_id}"

            components.append({
                "id": comp_id,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
            comp_id += 1

    return components


# =========================
# 自适应 eps（DBSCAN）
# =========================
def estimate_eps(points, k=4, q=0.6, min_eps=5.0, max_eps=30.0):
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    if n <= k:
        return min_eps

    knn_dists = []
    for i in range(n):
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        dists_sorted = np.sort(dists)
        knn_dists.append(dists_sorted[k])

    knn_dists = np.array(knn_dists)
    eps = float(np.quantile(knn_dists, q))
    return float(np.clip(eps, min_eps, max_eps))


def analyze_cluster_shape(cluster_pts):
    """
    对一个簇做简单的 PCA 形状分析（目前仅保留为可扩展，结果不参与后续逻辑）
    """
    if len(cluster_pts) < 3:
        return 1.0, len(cluster_pts)

    pts = np.array(cluster_pts, dtype=np.float32)
    center = pts.mean(axis=0, keepdims=True)
    pts_c = pts - center

    cov = np.cov(pts_c.T)
    vals, _vecs = np.linalg.eigh(cov)

    lam_min = max(vals[0], 1e-6)
    lam_max = max(vals[1], 1e-6)
    aspect_ratio = float(lam_max / lam_min)
    return aspect_ratio, len(cluster_pts)


def merge_points_adaptive(points,
                          base_eps=None,
                          k_for_eps=4,
                          q_for_eps=0.3,
                          min_eps=3.5,
                          max_eps=15,
                          min_samples=1):
    """
    最终点合并函数：自动估计 eps + DBSCAN + 每簇中值点 + 噪声点保留
    """
    if len(points) == 0:
        return []

    pts = np.array(points, dtype=np.float32)

    if base_eps is None:
        eps = estimate_eps(points,
                           k=k_for_eps,
                           q=q_for_eps,
                           min_eps=min_eps,
                           max_eps=max_eps)
    else:
        eps = float(base_eps)

    print(f"   [adaptive] eps={eps:.2f}, n_pts={len(points)}")

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_

    final_pts = []

    unique_labels = set(labels)
    for lb in unique_labels:
        if lb == -1:
            continue
        cluster_pts = pts[labels == lb]

        # 形状分析（目前只为可扩展，未参与决策）
        _aspect_ratio, _n_cluster = analyze_cluster_shape(cluster_pts)

        x_med = float(np.median(cluster_pts[:, 0]))
        y_med = float(np.median(cluster_pts[:, 1]))
        final_pts.append((x_med, y_med))

    # 噪声点直接保留
    for i, lb in enumerate(labels):
        if lb == -1:
            x, y = float(pts[i, 0]), float(pts[i, 1])
            final_pts.append((x, y))

    return final_pts


# =========================
# 点与元件 bbox 边的匹配：判定元件端点 / 电路节点
# =========================
def match_point_to_components(x, y, components, img_w, img_h):
    """
    对一个点 (x, y)，与所有元件 bbox 的 4 条边做几何检查。
    返回 matches: [{comp_id, cls_id, cls_name, side, edge_dist}, ...]
    side: "top" / "bottom" / "left" / "right"
    """
    matches = []

    # ---- 整图尺度因子：以 1200px 为基准，大图更宽容，小图不放大 ----
    min_hw = float(min(img_w, img_h))
    scale_img = min(max(min_hw / 1200.0, 1.0), 3.0)  # clamp 到 [1.0, 3.0]

    for comp in components:
        cid = comp["id"]
        cls_id = comp["cls_id"]
        cls_name = comp["cls_name"]
        x1, y1, x2, y2 = comp["bbox"]

        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        # ---- EDGE_EXTEND 自适应：元件越大、图越大，切向放宽越大 ----
        # 1) 元件尺度项：3% 的 max(bw,bh)
        ext_from_comp = 0.03 * max(bw, bh)

        # 2) 图像尺度项：短边的 0.2%（4000->8px，1000->2px）
        ext_from_img = 0.002 * min_hw

        extend = EDGE_EXTEND + ext_from_comp + ext_from_img
        extend = float(np.clip(extend, 5.0, 40.0))  # clamp：小图别太松，大图别太夸张


        # ---- 元件尺寸因子：元件越大，允许端点离边越远 ----
        edge_th = (EDGE_DIST_TH + 0.05 * max(bw, bh)) * scale_img
        edge_th = float(np.clip(edge_th, 8.0, 60.0))

        # ===== top edge (y = y1) =====
        # 切向约束：x 要落在 [x1-EDGE_EXTEND, x2+EDGE_EXTEND]
        if (x1 - extend) <= x <= (x2 + extend):
            dist_top = abs(y - y1)
            if dist_top <= edge_th:
                matches.append({
                    "comp_id": cid,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "side": "top",
                    "edge_dist": float(dist_top),
                })

        # ===== bottom edge (y = y2) =====
        if (x1 - extend) <= x <= (x2 + extend):
            dist_bottom = abs(y - y2)
            if dist_bottom <= edge_th:
                matches.append({
                    "comp_id": cid,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "side": "bottom",
                    "edge_dist": float(dist_bottom),
                })

        # ===== left edge (x = x1) =====
        # 切向约束：y 要落在 [y1-EDGE_EXTEND, y2+EDGE_EXTEND]
        if (y1 - extend) <= y <= (y2 + extend):
            dist_left = abs(x - x1)
            if dist_left <= edge_th:
                matches.append({
                    "comp_id": cid,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "side": "left",
                    "edge_dist": float(dist_left),
                })

        # ===== right edge (x = x2) =====
        if (y1 - extend) <= y <= (y2 + extend):
            dist_right = abs(x - x2)
            if dist_right <= edge_th:
                matches.append({
                    "comp_id": cid,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "side": "right",
                    "edge_dist": float(dist_right),
                })

    return matches

def classify_points(final_pts, components, img_w, img_h):
    """
    把最终合并后的点加上类型和匹配信息：
    - type: "component_terminal" / "node"
    - matches: list of {comp_id, cls_id, cls_name, side, edge_dist}
    """
    point_infos = []
    for idx, (x, y) in enumerate(final_pts):
        matches = match_point_to_components(x, y, components, img_w, img_h)
        # 只保留最可信的一条：edge_dist 最小
        # 这样阈值放宽后不会误把点匹配到多个元件上
        if matches:
            matches.sort(key=lambda m: m["edge_dist"])
            matches = [matches[0]]
        if len(matches) > 0:
            ptype = "component_terminal"
        else:
            ptype = "node"

        point_infos.append({
            "id": idx,
            "x": float(x),
            "y": float(y),
            "type": ptype,
            "matches": matches,
        })

    return point_infos


# =========================
# 可视化 & JSON 保存
# =========================
def visualize_final(img_path, point_infos, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot load image: {img_path}")
        return

    for p in point_infos:
        x = int(round(p["x"]))
        y = int(round(p["y"]))
        if p["type"] == "component_terminal":
            color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)  # green
        cv2.circle(img, (x, y), 4, color, -1)

    cv2.imwrite(save_path, img)
    print(f"[FINAL] Saved final merged points → {save_path}")


def save_points_json(name, point_infos, components, save_dir):
    js = {
        "image": name,
        "num_points": len(point_infos),
        "points": point_infos,
        "num_components": len(components),
        "components": components,
    }
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.json")
    with open(save_path, "w") as f:
        json.dump(js, f, indent=2)
    print(f"[JSON] Saved → {save_path}")


# =========================
# MAIN
# =========================
def main():
    json_list = sorted(glob(os.path.join(HAWP_JSON_DIR, "*.json")))

    for hawp_json in json_list:
        name = os.path.splitext(os.path.basename(hawp_json))[0]

        # 自动匹配所有可能的图像扩展名
        possible_exts = ["png", "jpg", "jpeg", "bmp"]
        img_path = None
        for ext in possible_exts:
            cand = os.path.join(IMG_DIR, f"{name}.{ext}")
            if os.path.exists(cand):
                img_path = cand
                break

        if img_path is None:
            print(f"[WARN] Missing image for {name} in {IMG_DIR}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        H, W = img.shape[:2]
        yolo_txt = os.path.join(YOLO_LABEL_DIR, f"{name}.txt")

        print(f"\n=== Processing {name} ===")

        # 1) 加载 HAWP 点 + YOLO junction 点
        hawp_pts = load_hawp_points(hawp_json)
        yolo_pts = load_yolo_junction_points(yolo_txt, W, H)

        # 2) 加载 YOLO 元件框（已按 clean_with_yolov9_lama.py 扩展）
        components = load_yolo_components(yolo_txt, W, H)

        # 3) 合并所有点并进行 DBSCAN 聚类
        all_pts = hawp_pts + yolo_pts
        min_hw = min(W, H)

        # ---- DBSCAN max_eps 自适应：大图放宽上限，小图仍保持合理范围 ----
        # 经验：min_hw*0.006 -> 1000->6, 2000->12, 4000->24
        max_eps_auto = int(round(min_hw * 0.006))
        max_eps_auto = int(np.clip(max_eps_auto, 15, 30))  # 最终上限 15~30

        final_pts = merge_points_adaptive(all_pts, max_eps=max_eps_auto)

        # 4) 判定元件端点 / 电路节点
        point_infos = classify_points(final_pts, components, W, H)

        # 5) 保存最终可视化和 JSON
        save_final_img = os.path.join(OUT_VIS, f"{name}_final_vis.png")
        visualize_final(img_path, point_infos, save_final_img)

        save_points_json(name, point_infos, components, OUT_JSON)


if __name__ == "__main__":
    main()
