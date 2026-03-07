# -*- coding: utf-8 -*-

import os
import json
import math
from glob import glob

import cv2
import numpy as np

from path_config import cfg_get, project_path

# ========= 基本路径 =========
MERGED_JSON_DIR = cfg_get("paths", "merged_points_json", default=project_path("outputs", "run1", "merged_points", "merged_points_json"))
LAMA_IMG_DIR    = cfg_get("paths", "suppressed_img", default=project_path("outputs", "run1", "suppressed_img"))
DAOXIAN_DIR     = cfg_get("paths", "dao_xian", default=project_path("outputs", "run1", "dao_xian"))

OUT_BASE   = cfg_get("paths", "link_root", default=project_path("outputs", "run1", "link"))
OUT_IMG    = os.path.join(OUT_BASE, "img")
OUT_JSON   = os.path.join(OUT_BASE, "json")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)

# ========= 超参数==========
MAX_SEARCH_DIST = None      # 自适应(按图像大小); 若设为数字则强制使用该半径
SEARCH_DIST_SCALE = 0.40  # max_dist = diag * scale
SEARCH_DIST_MIN   = 250.0 # 下限(小图也要能连上)
SEARCH_DIST_MAX   = 4000.0 # 上限(防止大图候选爆炸)
NEAR_K          = 120        # 每个点只看最近 K 个候选（提速 + 稳定）

CONNECT_RATIO_T = 0.7       # 连通阈值（想用0.95就改这里）
CONNECT_RADIUS  = 2          # 采样半径：斜线/细线/断裂更鲁棒（1~4都可试）

TERMINAL_MAX_DEGREE = 3      # terminal 一端最多3线
NODE_MAX_DEGREE     = 6      # node 最多 6 条

MIN_ANGLE_SEP_DEG   = 10.0   # ✅ node/terminal 上任意两条已连边的最小夹角（避免同向重复连）

# 可视化
DRAW_TEXT = True             # 是否画 ratio 文本
DRAW_RATIO_MIN = 0.7        # ratio 太小的不画（减少拥挤）

# ====== NEW: 修复断断续续的白线，让同一根线更容易成为一个连通域 ======
WIRE_REPAIR_FOR_CC = True

# 断裂桥接强度（自适应）：k = clamp(min(H,W)*ratio, k_min, k_max)
REPAIR_K_RATIO = 0.0020   # 例如 min=1000 -> k≈2；min=3000 -> k≈6
REPAIR_K_MIN   = 2
REPAIR_K_MAX   = 9
REPAIR_ITERS   = 1        # 1~2，断线多可改2，但别太大

# 是否做“方向性桥接”（对手绘折线/斜线更友好）
REPAIR_DIRECTIONAL = True


# -------------------------------------------------
# 工具函数：读取 merged_points.json
# -------------------------------------------------
def load_merged_json(json_path):
    with open(json_path, "r") as f:
        js = json.load(f)
    points = js.get("points", [])
    components = js.get("components", [])
    img_name = js.get("image", "")
    return img_name, points, components

def adaptive_ratio_th(dist, max_search_dist, base=0.82, min_th=0.55):
    """
    dist 越大，阈值越低（允许断线/间断）
    """
    if max_search_dist <= 1e-6:
        return base
    t = dist / float(max_search_dist)  # 0~1
    th = base - 0.22 * t               # 0.82 -> 0.60
    th = max(min_th, min(0.90, th))
    return float(th)

def repair_wire_mask_for_cc(wire_img):
    """
    只用于 CC/bridge 的 wire mask 修复：
    - closing: 连接小断裂
    - directional closing: 水平/垂直/斜向补断
    """
    if wire_img is None:
        return None
    H, W = wire_img.shape[:2]
    m = min(H, W)

    k = int(round(m * float(REPAIR_K_RATIO)))
    k = max(int(REPAIR_K_MIN), min(int(REPAIR_K_MAX), k))
    if k % 2 == 0:
        k += 1  # 让核大小为奇数更对称

    bin_img = (wire_img > 0).astype(np.uint8) * 255

    # 1) 普通 closing：补小断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=int(REPAIR_ITERS))

    # 2) 方向性 closing：对“轻笔折线/斜线”更有效，但不会像大圆核那样整体变粗
    if REPAIR_DIRECTIONAL and k >= 3:
        k2 = max(3, k)
        # 水平、垂直
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k2))

        out_h = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kh, iterations=1)
        out_v = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kv, iterations=1)

        # 两条对角线（用手工核）
        kd1 = np.eye(k2, dtype=np.uint8) * 255
        kd2 = np.fliplr(kd1)

        out_d1 = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kd1, iterations=1)
        out_d2 = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kd2, iterations=1)

        out = cv2.max(out_h, out_v)
        out = cv2.max(out, out_d1)
        out = cv2.max(out, out_d2)

    return out

# -------------------------------------------------
# 构建端点列表（保持原来的结构）
# -------------------------------------------------
def build_endpoints(points):
    endpoints = []
    terminal_indices = []
    node_indices = []
    eid = 0

    # 节点
    for p in points:
        if p.get("type") == "node":
            endpoints.append({
                "eid": eid,
                "kind": "node",
                "point_id": p["id"],
                "x": float(p["x"]),
                "y": float(p["y"]),
            })
            node_indices.append(eid)
            eid += 1

    # 元件端点
    for p in points:
        if p.get("type") != "component_terminal":
            continue
        px = float(p["x"])
        py = float(p["y"])
        for m in p.get("matches", []):
            endpoints.append({
                "eid": eid,
                "kind": "terminal",
                "point_id": p["id"],
                "x": px,
                "y": py,
                "comp_id": m["comp_id"],
                "cls_id": m["cls_id"],
                "cls_name": m["cls_name"],
                "side": m["side"],
                "edge_dist": float(m["edge_dist"]),
            })
            terminal_indices.append(eid)
            eid += 1

    return endpoints, terminal_indices, node_indices


# -------------------------------------------------
# 像素连通性检查：沿 A-B 直线采样，命中白线比例
# -------------------------------------------------
def sample_line_connectivity(wire_img, x0, y0, x1, y1, radius=1):
    h, w = wire_img.shape[:2]
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)

    if length < 1e-3:
        sx, sy = int(round(x0)), int(round(y0))
        if 0 <= sx < w and 0 <= sy < h and wire_img[sy, sx] > 0:
            return 1.0
        return 0.0

    num_samples = max(int(length), 2)
    hits = 0

    for i in range(num_samples + 1):
        t = i / num_samples
        x = x0 + dx * t
        y = y0 + dy * t
        sx = int(round(x))
        sy = int(round(y))

        found = False
        for yy in range(sy - radius, sy + radius + 1):
            if found:
                break
            for xx in range(sx - radius, sx + radius + 1):
                if 0 <= xx < w and 0 <= yy < h and wire_img[yy, xx] > 0:
                    hits += 1
                    found = True
                    break

    ratio = hits / (num_samples + 1)
    # 防御性：理论上 ratio 必须在 [0,1]
    if ratio > 1.0:
        ratio = 1.0
    return ratio




# =================================================
# NEW：连通域吸附 + “同连通域即连通” 判定
# 目的：解决 ①手绘弯线(直线采样ratio低) ②线很细 ③端点偏移
# =================================================
USE_CC_CONNECT = True          # 是否启用连通域连线判定（建议开）
CC_MIN_AREA = None            # 连通域最小面积阈值；None=按图像尺寸自适应
SNAP_MAX_R = 35               # 端点最大吸附半径（像素）
SNAP_STEP = 2                 # 扩圈步长
SNAP_PREFER_LARGE_ALPHA = 0.12  # 越大越偏向“大白色块”，防止吸到小噪声

# --- NEW: 连通域内“只保留必要边”（生成一棵生成树），避免同域内端点/节点两两乱连 ---
CC_BUILD_TREE = True             # 连通域内使用生成树连接（强烈建议 True）
CC_KNN = 24                      # 生成树候选边：每点只连最近 K 个（避免 O(n^2)）
CC_TERMINAL_MAX_DEGREE = 1       # 连通域生成树里 terminal 建议当叶子（最多 1 条边）
CC_NODE_MAX_DEGREE = None        # 连通域生成树里 node 不强制上限；如需可设 int

# --- NEW: 连通域生成树边的“像素支撑”筛选（提升美观：优先连有白像素支撑的局部边） ---
CC_TREE_RATIO_MIN = 0.30         # 同连通域内候选边的最小直线采样 ratio；太低则优先不选
CC_TREE_RATIO_RADIUS = 1         # 计算上述 ratio 的采样半径（建议 1~2）
CC_TREE_PREFER_HIGH_RATIO = True # 即使允许低 ratio，仍优先选更高 ratio 的边


# --- NEW: 不同连通域之间的“误连”抑制 ---
CROSS_CC_STRICT = True          # 若两点都在有效连通域但 label 不同：默认不允许直接用 ratio 连（防误连）
BRIDGE_DILATE_R = 2             # 允许用小膨胀尝试“补断线”连通性检查的膨胀半径（像素）
BRIDGE_MARGIN   = 12            # ROI 额外边距（像素）
BRIDGE_MAX_DIST = 1200.0        # 只对距离不太远的跨域候选做 bridge 检查（避免 ROI 过大）
BRIDGE_MAX_ROI_AREA = 900*900   # ROI 面积上限，超过则跳过 bridge 检查（防性能爆炸）


def compute_snap_max_r(wire_img):
    """端点吸附到白线连通域的最大半径：随图像短边变化，并做 clamp。"""
    H, W = wire_img.shape[:2]
    m = min(H, W)
    r = int(round(m * 0.012))   # 1000->12, 4000->48
    r = max(18, min(60, r))
    return int(r)

def _auto_cc_min_area(wire_img):
    """
    自适应连通域最小面积：
    - 既要过滤“孤立白噪点”，又不能误删细线段
    - 经验：按图像面积比例 + clamp
    """
    H, W = wire_img.shape[:2]
    area = H * W
    # 0.00001 * area： 2M像素 -> 20；12M像素 -> 120
    v = int(round(area * 1e-5))
    return int(max(20, min(1200, v)))

def build_wire_cc_index(wire_img, min_area=None):
    """
    返回：
      cc = {
        "labels": HxW int32 label map,
        "areas":  [num_labels] area,
        "keep":   [num_labels] bool
      }
    """
    if wire_img is None:
        return None
    bin_img = (wire_img > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)

    if min_area is None:
        min_area = _auto_cc_min_area(wire_img)

    keep = np.zeros((num_labels,), dtype=np.bool_)
    # 0 是背景不保留
    keep[1:] = areas[1:] >= int(min_area)

    return {
        "labels": labels,
        "areas": areas,
        "keep": keep,
        "min_area": int(min_area),
        "num_labels": int(num_labels),
    }

def snap_point_to_cc(cc, x, y, max_r=25, step=2, alpha=0.1):
    """
    把点 (x,y) 吸附到：附近“被保留的白色连通域(keep=True)”中的最近白像素。
    同时为了避免吸到小噪点：对候选像素所在连通域面积做加权偏好“大块”。

    返回：
      (sx, sy, label, dist)
      若找不到：返回 (x,y, 0, inf)
    """
    H, W = cc["labels"].shape[:2]
    x0 = int(round(x)); y0 = int(round(y))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))

    labels = cc["labels"]
    keep = cc["keep"]
    areas = cc["areas"]

    best = None  # (score, dist, sx, sy, lab)
    # 先检查自己是否已经在白连通域内
    lab0 = int(labels[y0, x0])
    if lab0 != 0 and keep[lab0]:
        return float(x0), float(y0), lab0, 0.0

    for r in range(1, int(max_r) + 1, int(max(1, step))):
        x1 = max(0, x0 - r); x2 = min(W, x0 + r + 1)
        y1 = max(0, y0 - r); y2 = min(H, y0 + r + 1)

        win = labels[y1:y2, x1:x2]
        # mask: keep 的白连通域
        # 注意：win 是 label map，不是 0/255
        mask = (win != 0)
        if mask.any():
            # 进一步只保留 keep 的 label
            win_lab = win[mask]
            win_keep = keep[win_lab]
            if not np.any(win_keep):
                continue

            # 取 keep 的像素坐标
            ys, xs = np.where(mask)
            ys = ys[win_keep]
            xs = xs[win_keep]
            labs = win_lab[win_keep]

            # 转回全图坐标
            xs_g = xs + x1
            ys_g = ys + y1

            # 计算距离
            dx = xs_g.astype(np.float32) - float(x0)
            dy = ys_g.astype(np.float32) - float(y0)
            d = np.sqrt(dx * dx + dy * dy)

            # 加权：更偏向大连通域（避免吸到小噪声）
            # score 越小越好
            a = areas[labs].astype(np.float32)
            score = d - float(alpha) * np.log1p(a)

            idx = int(np.argmin(score))
            cand = (float(score[idx]), float(d[idx]), float(xs_g[idx]), float(ys_g[idx]), int(labs[idx]))
            best = cand
            break

    if best is None:
        return float(x0), float(y0), 0, float("inf")
    _, dist, sx, sy, lab = best
    return sx, sy, lab, dist

def snap_endpoints_to_cc(endpoints, cc, max_r=25, step=2, alpha=0.1):
    """
    给 endpoints 增加/更新字段：
      - x_raw,y_raw: 原始点
      - x,y: 吸附后的点（用于后续连线判断）
      - cc_label: 所属连通域 label（0 表示没吸到）
      - snap_dist: 吸附距离
    """
    if cc is None:
        for e in endpoints:
            e["cc_label"] = 0
            e["snap_dist"] = None
        return endpoints

    for e in endpoints:
        if "x_raw" not in e:
            e["x_raw"] = float(e["x"])
            e["y_raw"] = float(e["y"])
        sx, sy, lab, dist = snap_point_to_cc(
            cc, e["x_raw"], e["y_raw"], max_r=max_r, step=step, alpha=alpha
        )
        e["x"] = float(sx)
        e["y"] = float(sy)
        e["cc_label"] = int(lab)
        e["snap_dist"] = float(dist) if math.isfinite(dist) else None
    return endpoints

def cc_connected(ep_a, ep_b):
    """两个端点若都吸附到同一个 keep 的连通域里，则视为强连通。"""
    la = int(ep_a.get("cc_label", 0))
    lb = int(ep_b.get("cc_label", 0))
    return (la != 0) and (la == lb)


def bridge_connected_by_dilate(wire_img, ax, ay, bx, by, dilate_r=2, margin=12, max_roi_area=810000):
    """
    在两点的包围盒 ROI 内对导线做轻微膨胀，然后检查两点是否在同一连通域。
    用途：处理“同一真实导线但因断裂导致分成两个 CC label”的情况。
    注意：只在 ROI 不大的时候用，避免性能开销。
    """
    H, W = wire_img.shape[:2]
    x1 = int(round(min(ax, bx))) - int(margin)
    y1 = int(round(min(ay, by))) - int(margin)
    x2 = int(round(max(ax, bx))) + int(margin)
    y2 = int(round(max(ay, by))) + int(margin)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    rw = int(x2 - x1); rh = int(y2 - y1)
    if rw <= 2 or rh <= 2:
        return False
    if rw * rh > int(max_roi_area):
        return False

    roi = (wire_img[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    if dilate_r and dilate_r > 0:
        k = int(2 * dilate_r + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        roi = cv2.dilate(roi, kernel, iterations=1)

    # ROI 坐标
    pax = int(round(ax)) - x1; pay = int(round(ay)) - y1
    pbx = int(round(bx)) - x1; pby = int(round(by)) - y1
    pax = max(0, min(rw - 1, pax)); pay = max(0, min(rh - 1, pay))
    pbx = max(0, min(rw - 1, pbx)); pby = max(0, min(rh - 1, pby))

    # 如果点不在白像素上，也允许：就从该点做 floodfill，看看能否到达对方附近
    # 为简单起见：connectedComponents 后比较 label
    num, labels, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
    la = int(labels[pay, pax])
    lb = int(labels[pby, pbx])
    if la == 0 or lb == 0:
        return False
    return la == lb

# -------------------------------------------------
# 角度工具：真实角度 + 最小夹角差（环形）
# -------------------------------------------------
def angle_deg(ax, ay, bx, by):
    """返回 A->B 的方向角 θ ∈ [0,360)"""
    return (math.degrees(math.atan2(by - ay, bx - ax)) + 360.0) % 360.0

def ang_diff_deg(a, b):
    """返回最小夹角差 ∈ [0,180]"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def angle_conflict(existing_angles, new_angle, min_sep):
    """若 new_angle 与任何已有角度夹角 < min_sep，则冲突"""
    for ea in existing_angles:
        if ang_diff_deg(ea, new_angle) < min_sep:
            return True
    return False

def compute_connect_radius_from_image(wire_img):
    H, W = wire_img.shape[:2]
    m = min(H, W)
    # 经验：小图 radius=1~2，大图 2~4
    r = int(round(m * 0.0015))  # 例如 1000px -> 2
    return int(max(1, min(4, r)))


# -------------------------------------------------
# NEW: 连通域内生成树（MST-like）— 只保留“连通所需的最少边”
# 目的：build_final_json 只需要连通分量来算 nets，所以同一连通域不需要两两相连。
# -------------------------------------------------
class _DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}
        self.r = {i: 0 for i in items}
        self.cnt = len(items)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        self.cnt -= 1
        return True

def _cc_max_deg(kind, term_max, node_max):
    if kind == "terminal":
        return term_max
    return node_max

def build_cc_spanning_edges(endpoints, wire_img):
    """
    为每个 cc_label 内的 endpoints 建一棵“生成树”(N-1 条边)，避免同域内两两乱连。
    NEW(美观优化)：
      - 优先选择“直线采样命中白像素比例(ratio)更高”的局部边
      - 第一轮：只用 ratio >= CC_TREE_RATIO_MIN 的候选边尝试连通
      - 若连不满（例如：弯折很厉害/或缺少中间 node），第二轮回退允许低 ratio，保证救回连通性
    """
    # label -> eids
    groups = {}
    ep_by = {int(e["eid"]): e for e in endpoints}
    for e in endpoints:
        lab = int(e.get("cc_label", 0))
        if lab == 0:
            continue
        groups.setdefault(lab, []).append(int(e["eid"]))

    out = []
    if wire_img is None:
        return out

    # ratio cache（避免重复采样）
    ratio_cache = {}

    def get_ratio(a, b):
        key = (a, b) if a < b else (b, a)
        if key in ratio_cache:
            return ratio_cache[key]
        ea = ep_by[a]; eb = ep_by[b]
        r = sample_line_connectivity(
            wire_img,
            float(ea["x"]), float(ea["y"]),
            float(eb["x"]), float(eb["y"]),
            radius=int(CC_TREE_RATIO_RADIUS)
        )
        ratio_cache[key] = float(r)
        return float(r)

    for lab, eids in groups.items():
        if len(eids) <= 1:
            continue

        # 建候选边：每点只取最近 CC_KNN 个（对大连通域提速）
        cand = {}  # (a,b) -> dist
        coords = [(eid, float(ep_by[eid]["x"]), float(ep_by[eid]["y"])) for eid in eids]
        for eid, x, y in coords:
            dlist = []
            for oeid, ox, oy in coords:
                if oeid == eid:
                    continue
                d = math.hypot(ox - x, oy - y)
                dlist.append((d, oeid))
            dlist.sort()
            k = int(min(CC_KNN, len(dlist)))
            for d, oeid in dlist[:k]:
                a, b = (eid, oeid) if eid < oeid else (oeid, eid)
                if (a, b) not in cand or d < cand[(a, b)]:
                    cand[(a, b)] = float(d)

        items = list(eids)

        def max_deg_local(kind, term_max, node_max):
            if kind == "terminal":
                return term_max
            return node_max

        def try_build(term_max, node_max, strict_ratio=True):
            dsu2 = _DSU(items)
            deg2 = {eid: 0 for eid in items}
            chosen = []

            # 组装候选列表并排序：
            # - strict_ratio=True：仅使用 ratio>=阈值的边，并按 dist 排序
            # - strict_ratio=False：允许低 ratio，但若 CC_TREE_PREFER_HIGH_RATIO=True，则优先更高 ratio
            eds = []
            for (a, b), d in cand.items():
                ra = get_ratio(a, b)
                if strict_ratio and (ra < float(CC_TREE_RATIO_MIN)):
                    continue
                eds.append((a, b, float(d), float(ra)))

            if strict_ratio:
                eds.sort(key=lambda t: t[2])  # dist
            else:
                if CC_TREE_PREFER_HIGH_RATIO:
                    eds.sort(key=lambda t: (-t[3], t[2]))  # ratio desc, dist asc
                else:
                    eds.sort(key=lambda t: t[2])

            for a, b, d, ra in eds:
                if dsu2.find(a) == dsu2.find(b):
                    continue

                ka = ep_by[a].get("kind")
                kb = ep_by[b].get("kind")
                ma = max_deg_local(ka, term_max, node_max)
                mb = max_deg_local(kb, term_max, node_max)

                if ma is not None and deg2[a] >= int(ma):
                    continue
                if mb is not None and deg2[b] >= int(mb):
                    continue

                if dsu2.union(a, b):
                    deg2[a] += 1
                    deg2[b] += 1
                    chosen.append((a, b, d, ra))
                    if len(chosen) >= len(items) - 1:
                        break

            ok = (dsu2.cnt == 1) or (len(items) == 0)
            return ok, chosen

        # ① 严格 ratio（更美观：尽量连“真有白像素支撑”的局部边）
        ok, chosen = try_build(CC_TERMINAL_MAX_DEGREE, CC_NODE_MAX_DEGREE, strict_ratio=True)

        # ② 若 terminal 叶子约束导致连不满，先放宽 terminal 约束，但仍严格 ratio
        if not ok:
            ok, chosen = try_build(term_max=999999, node_max=CC_NODE_MAX_DEGREE, strict_ratio=True)

        # ③ 若仍连不满（典型：弯折太大/中间 node 缺失），回退允许低 ratio（保证“救回连通性”）
        if not ok:
            ok, chosen = try_build(term_max=999999, node_max=CC_NODE_MAX_DEGREE, strict_ratio=False)

        for a, b, d, ra in chosen:
            out.append({
                "eid1": int(a),
                "eid2": int(b),
                "phase": 0,
                "dir": "cc_tree",
                "dist": float(d),
                # 用真实 ratio 画出来，方便看“美观边” vs “救回边”
                "ratio": float(ra),
                "cc_label": int(lab),
            })

    return out

# -------------------------------------------------
# NEW：按距离从近到远 + ratio 连接 + node 角度间隔约束
# -------------------------------------------------
def connect_by_nearest(endpoints, wire_img, max_search_dist, base_ratio=None):
    if base_ratio is None:
        base_ratio = float(CONNECT_RATIO_T)
    # ---- 自适应角度间隔：候选越多/点越密 -> 角度约束更强；反之更弱
    k = int(NEAR_K)
    min_sep = 8.0 + 6.0 * (k / 260.0)     # k=60 -> ~9.4°, k=260 -> 14°
    min_sep = float(max(8.0, min(16.0, min_sep)))

    ep = {e["eid"]: e for e in endpoints}

    # 度数限制
    deg = {e["eid"]: 0 for e in endpoints}

    # ✅ node 已连边角度列表（连续角度，不量化）
    used_angles = {e["eid"]: [] for e in endpoints if e["kind"] in ("node","terminal")}

    # 去重边
    edge_pairs = set()
    edges = []

    # 预提坐标
    coords = [(e["eid"], e["x"], e["y"], e["kind"]) for e in endpoints]

    def max_deg(kind):
        return TERMINAL_MAX_DEGREE if kind == "terminal" else NODE_MAX_DEGREE

    for (eid, ax, ay, akind) in coords:
        if deg[eid] >= max_deg(akind):
            continue

        # 候选：只取半径内，并按距离排序，截断 NEAR_K
        cand = []
        for (oeid, bx, by, bkind) in coords:
            if oeid == eid:
                continue
            d = math.hypot(bx - ax, by - ay)
            if d < 1:
                continue
            if d > max_search_dist:
                continue
            cand.append((d, oeid))

        cand.sort()
        if NEAR_K > 0:
            cand = cand[:NEAR_K]

        # 从近到远尝试
        for dist, oeid in cand:
            if deg[eid] >= max_deg(akind):
                break

            bkind = ep[oeid]["kind"]
            if deg[oeid] >= max_deg(bkind):
                continue

            a, b = (eid, oeid) if eid < oeid else (oeid, eid)
            if (a, b) in edge_pairs:
                continue

            # ✅ 角度间隔冲突检查（约束 node + component terminal）
            th_a = angle_deg(ax, ay, ep[oeid]["x"], ep[oeid]["y"])  # A->B
            th_b = (th_a + 180.0) % 360.0                           # B->A

            if akind in ("node","terminal"):
                if angle_conflict(used_angles[eid], th_a, min_sep):
                    continue
            if bkind in ("node","terminal"):
                if angle_conflict(used_angles[oeid], th_b, min_sep):
                    continue

            # 连通率裁决（主判据）
            # 1) ✅ 连通域判定：若两点吸附到同一白色连通域，则直接认为连通
            if USE_CC_CONNECT and CC_BUILD_TREE and cc_connected(ep[eid], ep[oeid]):
                # 同一连通域内的连线交给 cc_tree 统一生成（避免两两乱连）
                continue
            else:
                # 2) 回退：两点不在同一连通域
                #    - 若两点都吸附到了“有效连通域”但 label 不同：优先防止误连
                la = int(ep[eid].get("cc_label", 0))
                lb = int(ep[oeid].get("cc_label", 0))
                if CROSS_CC_STRICT and (la != 0) and (lb != 0) and (la != lb):
                    # 允许做一次“轻微膨胀 + ROI 连通性”检查，修复因断线导致的 CC 分裂
                    if float(dist) <= float(BRIDGE_MAX_DIST):
                        ok = bridge_connected_by_dilate(
                            wire_img, ax, ay, ep[oeid]["x"], ep[oeid]["y"],
                            dilate_r=BRIDGE_DILATE_R, margin=BRIDGE_MARGIN,
                            max_roi_area=BRIDGE_MAX_ROI_AREA
                        )
                        if ok:
                            ratio = 1.5  # 标记为 bridge 连接（可视化时会显示 1.50）
                            dir_name = "bridge"
                        else:
                            continue
                    else:
                        continue
                else:
                    # 正常回退：沿 A-B 直线采样命中白像素比例（原逻辑）
                    ratio = sample_line_connectivity(
                        wire_img, ax, ay, ep[oeid]["x"], ep[oeid]["y"], radius=CONNECT_RADIUS
                    )
                    ratio_th = adaptive_ratio_th(dist, max_search_dist, base=base_ratio)
                    if ratio < ratio_th:
                        continue
                    dir_name = "nearest"


            # 建边：字段保持和旧版一致（phase/dir/dist/ratio 都有）
            edges.append({
                "eid1": eid,
                "eid2": oeid,
                "phase": 0,            # 新逻辑统一标 0
                "dir": dir_name,      # 不再是 left/right/up/down
                "dist": float(dist),
                "ratio": float(ratio),
            })
            edge_pairs.add((a, b))

            # 更新度数
            deg[eid] += 1
            deg[oeid] += 1

            # ✅ 更新已用角度（连续角度）
            if akind in ("node","terminal"):
                used_angles[eid].append(th_a)
            if bkind in ("node","terminal"):
                used_angles[oeid].append(th_b)

            # terminal 一端一线：连上就停
            if akind == "terminal":
                break

    return edges


# -------------------------------------------------
# 可视化（画点/画边/写 ratio）
# -------------------------------------------------
def visualize_graph(name, lama_img, endpoints, edges, save_path):
    img = lama_img.copy()
    ep_dict = {ep["eid"]: ep for ep in endpoints}

    # 点
    for ep in endpoints:
        x, y = int(ep["x"]), int(ep["y"])
        color = (0, 0, 255) if ep["kind"] == "terminal" else (0, 255, 0)
        cv2.circle(img, (x, y), 4, color, -1)

    # 边
    for e in edges:
        a = ep_dict[e["eid1"]]
        b = ep_dict[e["eid2"]]
        x1, y1 = int(a["x"]), int(a["y"])
        x2, y2 = int(b["x"]), int(b["y"])

        # 颜色：terminal-terminal / terminal-node / node-node
        if a["kind"] == "terminal" and b["kind"] == "terminal":
            color = (255, 0, 255)
        elif "terminal" in (a["kind"], b["kind"]):
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)

        cv2.line(img, (x1, y1), (x2, y2), color, 2)

        # ratio 文本
        if DRAW_TEXT:
            ratio = float(e.get("ratio", 0.0))
            if ratio >= DRAW_RATIO_MIN:
                mx = int((x1 + x2) * 0.5)
                my = int((y1 + y2) * 0.5)
                cv2.putText(
                    img, f"{ratio:.2f}",
                    (mx + 3, my + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

    cv2.imwrite(save_path, img)
    print("[VIS] Saved", save_path)


# -------------------------------------------------
# JSON 输出（沿用旧结构）
# -------------------------------------------------
def save_graph_json(name, points, components, endpoints, edges, save_path):
    js = {
        "image": name,
        "num_points": len(points),
        "points": points,
        "num_components": len(components),
        "components": components,
        "num_endpoints": len(endpoints),
        "endpoints": endpoints,
        "num_edges": len(edges),
        "edges": edges,
    }
    with open(save_path, "w") as f:
        json.dump(js, f, indent=2)
    print("[JSON] Saved", save_path)




def compute_max_search_dist_from_image(img):
    """根据图像尺寸自适应搜索半径：diag * SEARCH_DIST_SCALE，并做上下限截断。"""
    h, w = img.shape[:2]
    diag = math.hypot(w, h)
    msd = diag * float(SEARCH_DIST_SCALE)
    msd = max(float(SEARCH_DIST_MIN), min(float(SEARCH_DIST_MAX), msd))
    return float(msd)

def find_image(base_dir, name):
    """
    自动匹配实际存在的图像文件：
    支持 {name}.png / {name}.jpg / {name}_final.png / {name}_final.jpg
    """
    exts = ["png", "jpg", "jpeg", "bmp"]
    suffixes = ["", "_final"]

    for suf in suffixes:
        for ext in exts:
            cand = os.path.join(base_dir, f"{name}{suf}.{ext}")
            if os.path.exists(cand):
                return cand
    return None

def compute_near_k(endpoints, wire_img, max_search_dist):
    H, W = wire_img.shape[:2]
    N = max(1, len(endpoints))
    area = float(H * W)
    density = N / max(1.0, area)  # endpoints per pixel

    # 期望邻居数 ~ density * pi * r^2
    exp_neighbors = density * math.pi * (float(max_search_dist) ** 2)

    # 给一点余量（避免截断太狠）
    k = int(round(exp_neighbors * 1.5))

    # clamp：别太小也别爆炸
    k = max(60, min(260, k))
    return int(k)
# -------------------------------------------------
# 主函数
# -------------------------------------------------
def main():
    global CONNECT_RADIUS, NEAR_K
    json_list = sorted(glob(os.path.join(MERGED_JSON_DIR, "*.json")))
    if not json_list:
        print("[WARN] no json found")
        return

    print("==== build_connections (nearest-first + angle-gap) ====")
    print(f"[CFG] MAX_SEARCH_DIST={MAX_SEARCH_DIST} (None=auto), SEARCH_DIST_SCALE={SEARCH_DIST_SCALE}, SEARCH_DIST_MIN={SEARCH_DIST_MIN}, SEARCH_DIST_MAX={SEARCH_DIST_MAX}, NEAR_K={NEAR_K}")
    print(f"[CFG] CONNECT_RATIO_T={CONNECT_RATIO_T}, CONNECT_RADIUS={CONNECT_RADIUS}")
    print(f"[CFG] TERMINAL_MAX_DEGREE={TERMINAL_MAX_DEGREE}, NODE_MAX_DEGREE={NODE_MAX_DEGREE}")
    print(f"[CFG] MIN_ANGLE_SEP_DEG={MIN_ANGLE_SEP_DEG}")

    for jpath in json_list:
        name = os.path.splitext(os.path.basename(jpath))[0]

        # 背景图（仅用于可视化）
        img_path = find_image(DAOXIAN_DIR, name)
        lama_img = cv2.imread(img_path) if img_path else None
        if lama_img is None:
            print("[WARN] missing bg img", img_path)
            continue

        # 导线图（remove_components 输出）
        wire_path = os.path.join(DAOXIAN_DIR, f"{name}_final.png")
        wire_img = cv2.imread(wire_path, cv2.IMREAD_GRAYSCALE)
        if wire_img is None:
            print("[WARN] missing wire img", wire_path)
            continue

        wire_density = float(np.mean(wire_img > 0))
        base_ratio = 0.62 + 0.20 * min(1.0, wire_density / 0.02)
        base_ratio = float(max(0.60, min(0.82, base_ratio)))

        # 自适应搜索半径（也可把 MAX_SEARCH_DIST 设成数字来强制）
        auto_msd = compute_max_search_dist_from_image(wire_img)
        max_search_dist = float(MAX_SEARCH_DIST) if isinstance(MAX_SEARCH_DIST, (int, float)) else auto_msd
        print(f"[CFG@{name}] max_search_dist={max_search_dist:.1f} (auto={auto_msd:.1f})")

        print(f"\n=== Processing {name} ===")
        _, points, comps = load_merged_json(jpath)

        endpoints, _, _ = build_endpoints(points)

        # ===== NEW: 端点吸附到白线连通域（减少“点偏/弯线/细线”导致的漏连）=====
        wire_img_for_cc = wire_img
        if WIRE_REPAIR_FOR_CC:
            wire_img_for_cc = repair_wire_mask_for_cc(wire_img)

        cc = build_wire_cc_index(wire_img_for_cc, CC_MIN_AREA) if USE_CC_CONNECT else None
        if cc is not None:
            print(f"[CC] num_labels={cc['num_labels']} keep_min_area={cc['min_area']}")
        snap_r = compute_snap_max_r(wire_img_for_cc)
        snap_endpoints_to_cc(
            endpoints, cc,
            max_r=snap_r, step=SNAP_STEP, alpha=SNAP_PREFER_LARGE_ALPHA
        )

        auto_r = compute_connect_radius_from_image(wire_img)
        auto_k = compute_near_k(endpoints, wire_img, max_search_dist)
        CONNECT_RADIUS = auto_r
        NEAR_K = auto_k
        print(f"[AUTO] NEAR_K={NEAR_K} base_ratio={base_ratio:.2f}")

        cc_tree_edges = build_cc_spanning_edges(endpoints, wire_img_for_cc) if (USE_CC_CONNECT and CC_BUILD_TREE) else []
        edges = connect_by_nearest(endpoints, wire_img, max_search_dist, base_ratio=base_ratio)
        edges = cc_tree_edges + edges
        print(f"[EDGE] total edges = {len(edges)}")

        # 可视化
        save_img = os.path.join(OUT_IMG, f"{name}_connections.png")
        visualize_graph(name, lama_img, endpoints, edges, save_img)

        # JSON
        save_json = os.path.join(OUT_JSON, f"{name}_graph.json")
        save_graph_json(name, points, comps, endpoints, edges, save_json)

if __name__ == "__main__":
    main()
