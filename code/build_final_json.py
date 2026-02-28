# -*- coding: utf-8 -*-
"""
build_final_json.py

把：
  1) link/json/cir*_graph.json  (build_connections.py 输出的点+端点+边)
  2) fuse_json/cir*.json        (fuse_yolo_parseq.py 输出的文本-元件匹配)

融合成：
  final_result/json/{name}.final.json

并输出：
  final_result/img/{name}_final.png  综合可视化
"""

import os
import json
from glob import glob
import math

import cv2
import numpy as np

# ----------- 路径配置 ------------
GRAPH_JSON_DIR = "/root/autodl-tmp/final_result/link/json"
FUSE_JSON_DIR  = "/root/autodl-tmp/final_result/fuse_json"
LAMA_IMG_DIR   = "/root/autodl-tmp/final_result/src_img"
TYPE_REFINE_DIR = "/root/autodl-tmp/final_result/type_refine/json"
PORTS_PATCH_DIR = "/root/autodl-tmp/final_result/ports_cls/json"

OUT_BASE = "/root/autodl-tmp/final_result/final_result"
OUT_IMG  = os.path.join(OUT_BASE, "img")
OUT_JSON = os.path.join(OUT_BASE, "json")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_JSON, exist_ok=True)


# -------------------------------------------------
# 工具：加载 graph_json (build_connections.py 的输出)
# -------------------------------------------------
def load_graph_json(path):
    with open(path, "r") as f:
        js = json.load(f)
    # 结构参考 build_connections.py: save_graph_json :contentReference[oaicite:2]{index=2}
    image       = js.get("image", "")
    points      = js.get("points", [])
    components  = js.get("components", [])
    endpoints   = js.get("endpoints", [])
    edges       = js.get("edges", [])
    return image, points, components, endpoints, edges


# -------------------------------------------------
# 工具：加载 fuse_json (fuse_yolo_parseq 输出)
# 结构：list[ {image, text, text_box[4], match_relation, matched_component_id, ...} ]
# -------------------------------------------------
def load_fuse_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        js = json.load(f)
    # 直接返回 list
    return js

def resolve_texts_to_component_instances(texts, components, img_w, img_h):
    """
    将 fuse_json 输出的 “类别级近邻”(matched_component_cls_id) 绑定到
    最终 graph_json 的 “实例级组件”(components[].id)

    texts: list[dict] from fuse_json
    components: list[dict] from graph_json (each has id, cls_id, cls_name, bbox=[x1,y1,x2,y2])
    img_w/img_h: from lama image
    """
    if not texts or not components or not img_w or not img_h:
        return texts

    # 预计算 component centers
    comp_centers = []
    for c in components:
        bbox = c.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        comp_centers.append((c, cx, cy))

    if not comp_centers:
        return texts

    out = []
    for t in texts:
        # 只处理 component 关系
        if t.get("match_relation") != "component":
            out.append(t)
            continue

        tb = t.get("text_box", None)
        if not tb or len(tb) != 4:
            out.append(t)
            continue

        xc, yc, bw, bh = tb
        tx = float(xc) * float(img_w)
        ty = float(yc) * float(img_h)

        want_cls = t.get("matched_component_cls_id", None)
        if want_cls is not None:
            try:
                want_cls = int(want_cls)
            except Exception:
                want_cls = None

        best = None
        best_d2 = 1e30

        for c, cx, cy in comp_centers:
            if want_cls is not None and int(c.get("cls_id", -999)) != want_cls:
                continue
            dx = tx - cx
            dy = ty - cy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = c

        # 如果按 cls 过滤找不到（比如 fuse 里 cls_id=None 或该类没检出），降级为全体最近
        if best is None:
            for c, cx, cy in comp_centers:
                dx = tx - cx
                dy = ty - cy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = c

        t2 = dict(t)
        if best is not None:
            t2["matched_component_id"] = int(best.get("id"))
            t2["component_name"] = best.get("cls_name")
            # 保留/补齐 cls 信息
            if t2.get("matched_component_cls_id") is None:
                t2["matched_component_cls_id"] = int(best.get("cls_id"))
            if t2.get("component_cls_name") is None:
                t2["component_cls_name"] = best.get("cls_name")
            # 可选：记录像素距离（方便 debug）
            t2["instance_match_dist_px"] = float(math.sqrt(best_d2))
        out.append(t2)

    return out

# -------------------------------------------------
# 工具：构建 crossover 额外连线
# -------------------------------------------------
def build_crossover_edges(endpoints):
    """
    输入：graph_json 里的 endpoints
    输出：专门为 crossover 加的“竖向/横向”边列表

    逻辑：
      对每一个 cls_name == "crossover" 的元件：
        - top / bottom 端点  → 竖向网络
        - left / right 端点  → 横向网络
      只连最近的一对 (top-bottom, left-right)，防止一堆重复线
    """
    ep_dict = {ep["eid"]: ep for ep in endpoints}

    # 统计：per crossover component
    # comp_id -> { "top":[eid...], "bottom":[...], "left":[...], "right":[...] }
    cross_by_comp = {}

    for ep in endpoints:
        if ep.get("kind") != "terminal":
            continue
        cls_name = ep.get("cls_name", "")
        if cls_name != "crossover":
            continue

        comp_id = ep.get("comp_id")
        side    = ep.get("side")
        if comp_id is None or side is None:
            continue

        bucket = cross_by_comp.setdefault(comp_id, {
            "top": [], "bottom": [], "left": [], "right": []
        })
        if side in bucket:
            bucket[side].append(ep["eid"])

    extra_edges = []

    def _euclidean(eid1, eid2):
        a = ep_dict[eid1]
        b = ep_dict[eid2]
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    # 对每个 crossover 元件，构建上下/左右连线
    for comp_id, sides in cross_by_comp.items():
        tops    = sides["top"]
        bottoms = sides["bottom"]
        lefts   = sides["left"]
        rights  = sides["right"]

        # ---------- 竖向：top-bottom ----------
        if tops and bottoms:
            best_pair = None
            best_dist = 1e18
            for t_eid in tops:
                for b_eid in bottoms:
                    d = _euclidean(t_eid, b_eid)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (t_eid, b_eid)
            if best_pair is not None:
                t_eid, b_eid = best_pair
                extra_edges.append({
                    "eid1": t_eid,
                    "eid2": b_eid,
                    "phase": 3,                 # 特殊标记，不和原 phase1/2 冲突
                    "dir": "crossover_v",       # v = vertical
                    "dist": best_dist,
                    "ratio": 3.0,               # 直接给满分连通
                })

        # ---------- 横向：left-right ----------
        if lefts and rights:
            best_pair = None
            best_dist = 1e18
            for l_eid in lefts:
                for r_eid in rights:
                    d = _euclidean(l_eid, r_eid)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (l_eid, r_eid)
            if best_pair is not None:
                l_eid, r_eid = best_pair
                extra_edges.append({
                    "eid1": l_eid,
                    "eid2": r_eid,
                    "phase": 3,
                    "dir": "crossover_h",       # h = horizontal
                    "dist": best_dist,
                    "ratio": 3.0,
                })

    return extra_edges


# -------------------------------------------------
# 工具：根据 edges 计算网络（连通分量）
# -------------------------------------------------
def build_nets(endpoints, edges):
    """
    根据所有边（包括 crossover 补线）做一个简单的连通分量，
    每个 net 是一组 eid。

    返回：
      nets = [
        {
          "id": 0,
          "endpoint_eids": [...],
          "component_ids": [...],     # 出现在该网络上的元件 id（terminal 的 comp_id）
        },
        ...
      ]
    """
    ep_dict = {ep["eid"]: ep for ep in endpoints}

    # adjacency
    adj = {}
    for ep in endpoints:
        adj[ep["eid"]] = set()
    for e in edges:
        a = e["eid1"]
        b = e["eid2"]
        if a not in adj or b not in adj:
            continue
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    nets = []
    net_id = 0

    for eid_start in adj.keys():
        if eid_start in visited:
            continue

        # BFS
        stack = [eid_start]
        comp_eids = []
        visited.add(eid_start)

        while stack:
            cur = stack.pop()
            comp_eids.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        # 统计网络上的元件 id（只看 terminal）
        comp_ids = set()
        for eid in comp_eids:
            ep = ep_dict[eid]
            if ep.get("kind") == "terminal":
                cid = ep.get("comp_id")
                if cid is not None:
                    comp_ids.add(cid)

        nets.append({
            "id": net_id,
            "endpoint_eids": sorted(comp_eids),
            "component_ids": sorted(list(comp_ids)),
        })
        net_id += 1

    return nets


# -------------------------------------------------
# 可视化：整幅电路图信息
# -------------------------------------------------
from PIL import Image, ImageDraw, ImageFont

def visualize_full(
    name,
    lama_img,
    points,
    components,
    endpoints,
    edges,
    texts,
    save_path,
):
    """
    使用 Pillow 绘制 Unicode（Ω µ ° ± 等）
    不依赖 freetype，不下载字体，不访问网络。
    """

    h, w = lama_img.shape[:2]

    # 转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(lama_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # ===============================
    #  只使用本地字体（不会网络下载）
    # ===============================
    font_path = "/root/autodl-tmp/final_result/font/DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"字体不存在: {font_path}\n请执行:\n"
            "mkdir -p /root/autodl-tmp/final_result/font\n"
            "cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /root/autodl-tmp/final_result/font/"
        )

    font = ImageFont.truetype(font_path, 22)

    # ------------------------------
    # 绘制元件
    # ------------------------------
    for comp in components:
        bbox = comp.get("bbox", None)
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 200, 0), width=2)

        cls0 = comp.get("cls_name", f"id={comp.get('id')}")
        cls1 = comp.get("cls_name_refined", None)
        subt = comp.get("subtype", None)

        if cls1 and cls1 != cls0:
            # 有细分类：同时显示大类 + 细类（可选把 subtype 也挂上）
            if subt:
                txt = f"{cls0} | {cls1} ({subt})"
            else:
                txt = f"{cls0} | {cls1}"
        else:
            # 无细分类或细分类等于大类：只显示一个
            if subt:
                txt = f"{cls0} ({subt})"
            else:
                txt = cls0

        draw.text((x1, max(0, y1 - 20)), txt, font=font, fill=(0, 200, 0))

    # ------------------------------
    # 绘制 fuse 文本框（xywh-center 格式）
    # ------------------------------
    for t in texts:
        tb = t.get("text_box")
        if not tb:
            continue

        xc, yc, bw, bh = tb

        # 转回像素：注意 xc,yc 是中心点！
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        draw.rectangle([(x1, y1), (x2, y2)], outline=(200, 0, 0), width=2)

        txt_label = t.get("text", "")
        draw.text((x1, y2 + 5), txt_label, font=font, fill=(200, 0, 0))


    # ------------------------------
    # 绘制节点与端点（支持 ViTPose 端点角色标注 + 与 HAWP 点合并显示）
    # ------------------------------
    ep_dict = {ep["eid"]: ep for ep in endpoints}

    # 1) 读取 ports_patch.json，拿到 component_ports（含 vitpose kps 和 matched_eid）
    comp_ports = {}
    ports_patch_path = os.path.join(PORTS_PATCH_DIR, f"{name}_ports_patch.json")
    if os.path.exists(ports_patch_path):
        try:
            with open(ports_patch_path, "r") as f:
                ports_js = json.load(f)
            comp_ports = ports_js.get("component_ports", {}) or {}
        except Exception as e:
            print(f"[WARN] cannot load ports_patch for vis: {ports_patch_path} ({e})")
            comp_ports = {}

    # 2) 统计：哪些 HAWP eid 被 vitpose 匹配到了 —— 只用于“在 HAWP 点旁边标注角色”
    matched_hawp_eids = set()

    # eid -> list of role strings（可能一个 eid 被多个 role 命中；一般不会，但做个保护）
    eid2roles = {}

    # 同时收集：未匹配的 vitpose 点（直接画出来）
    vitpose_unmatched = []  # list of (x, y, role, conf, comp_id)

    for cid_str, info in comp_ports.items():
        kps = info.get("kps", []) or []
        for kp in kps:
            role = kp.get("role", "")
            vx = kp.get("x", None)
            vy = kp.get("y", None)
            conf = kp.get("conf", None)
            me = kp.get("matched_eid", None)

            if vx is None or vy is None:
                continue

            if me is not None and me in ep_dict:
                matched_hawp_eids.add(me)
                eid2roles.setdefault(me, []).append(role)
            else:
                vitpose_unmatched.append((float(vx), float(vy), role, conf, cid_str))

    # 3) 画 HAWP endpoints：永远用 HAWP 原坐标
    for ep in endpoints:
        eid = ep["eid"]
        x, y = int(ep["x"]), int(ep["y"])

        if ep["kind"] == "terminal":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=color)

        # 3.1) 如果这个 HAWP 点被 vitpose 匹配到了：在它旁边标注 vitpose 角色
        if eid in matched_hawp_eids:
            roles = eid2roles.get(eid, [])
            # 去重并稳定顺序
            roles = list(dict.fromkeys([r for r in roles if r]))
            if roles:
                role_txt = "/".join(roles)
                draw.text((x + 4, max(0, y - 18)), role_txt, font=font, fill=(255, 140, 0))

        # 3.2) endpoint 自带的 port_role（可选）
        # 只在“该 HAWP 点没有被 vitpose matched_eid 命中”时才画，避免重复显示
        if ep.get("port_role", None) and (eid not in matched_hawp_eids):
            draw.text((x + 4, max(0, y + 2)), str(ep["port_role"]), font=font, fill=(255, 0, 0))

    # 4) 画 vitpose “未匹配”的端点：用不同形状/颜色（蓝色小方块）
    for (vx, vy, role, conf, cid_str) in vitpose_unmatched:
        x, y = int(vx), int(vy)
        draw.rectangle([(x - 4, y - 4), (x + 4, y + 4)], outline=(30, 144, 255), width=2)
        draw.text((x + 4, max(0, y - 18)), f"{role}", font=font, fill=(30, 144, 255))

    # ------------------------------
    # 绘制连接线：永远使用 HAWP 原坐标（不做中点合并）
    # ------------------------------
    for e in edges:
        a_id = e["eid1"]
        b_id = e["eid2"]
        if a_id not in ep_dict or b_id not in ep_dict:
            continue

        x1, y1 = int(ep_dict[a_id]["x"]), int(ep_dict[a_id]["y"])
        x2, y2 = int(ep_dict[b_id]["x"]), int(ep_dict[b_id]["y"])

        dir_name = e.get("dir", "")
        a = ep_dict[a_id]
        b = ep_dict[b_id]

        if isinstance(dir_name, str) and dir_name.startswith("crossover_"):
            color = (255, 255, 0)
            width = 3
        else:
            if a["kind"] == "terminal" and b["kind"] == "terminal":
                color = (255, 0, 255)
            elif "terminal" in (a["kind"], b["kind"]):
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            width = 2

        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)


    # 保存
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_cv)
    print("[VIS-FINAL] Saved:", save_path)

def find_image(base_dir, name):
    exts = ["png", "jpg", "jpeg", "bmp"]
    suffixes = ["", "_final"]
    for suf in suffixes:
        for ext in exts:
            cand = os.path.join(base_dir, f"{name}{suf}.{ext}")
            if os.path.exists(cand):
                return cand
    return None


# -------------------------------------------------
# ✅ 新增：合并 type refine / ports patch 到最终结构
#   - 保留 components[*].cls_name
#   - 新增 components[*].cls_name_refined
# -------------------------------------------------
def apply_type_refine_patch(base_name, components):
    """
    从 {TYPE_REFINE_DIR}/json/{base}_type_refine.json 读取补丁，
    将补丁字段合并到 components 中（不覆盖 cls_name）。
    """
    patch_path = os.path.join(TYPE_REFINE_DIR, f"{base_name}_type_refine.json")
    if not os.path.exists(patch_path):
        # 没有细分类结果也正常
        return

    try:
        with open(patch_path, "r") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] cannot load type refine patch: {patch_path} ({e})")
        return

    patch_by_id = js.get("patch_by_component_id", {}) or {}
    applied = 0

    for comp in components:
        cid = comp.get("id", None)
        if cid is None:
            continue
        patch = patch_by_id.get(str(cid))
        if not patch:
            continue

        # ⭐字段规范：保留 cls_name，新增 cls_name_refined
        if "cls_name_refined" in patch and patch["cls_name_refined"] is not None:
            comp["cls_name_refined"] = patch["cls_name_refined"]

        # 其余可选信息（不强制，但保留有助于 debug / 论文统计）
        for k in [
            "subtype",
            "subtype_conf",
            "variant",
            "variant_conf",
            "method",
            "topk",
            "matched_terminal_count",
            "expected_ports",
            "used_ports",
            "note",
        ]:
            if k in patch and patch[k] is not None:
                comp[k] = patch[k]

        applied += 1

    print(f"  -> type_refine patch applied: {applied}/{len(components)}")


def apply_ports_patch(base_name, endpoints):
    """
    从 {PORTS_PATCH_DIR}/json/{base}_ports_patch.json 读取补丁，
    将端点角色/置信度等信息合并到 endpoints 中。
    """
    patch_path = os.path.join(PORTS_PATCH_DIR, f"{base_name}_ports_patch.json")
    if not os.path.exists(patch_path):
        return

    try:
        with open(patch_path, "r") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] cannot load ports patch: {patch_path} ({e})")
        return

    role_patch = js.get("endpoint_role_patch", {}) or {}
    applied = 0

    for ep in endpoints:
        eid = ep.get("eid", ep.get("id", None))
        if eid is None:
            continue
        patch = role_patch.get(str(eid))
        if not patch:
            continue

        # 只写入你输出的字段；若已有同名字段，会以 patch 为准
        for k in ["port_role", "port_conf", "port_from", "comp_id"]:
            if k in patch and patch[k] is not None:
                ep[k] = patch[k]

        applied += 1

    print(f"  -> ports_patch applied: {applied}/{len(endpoints)}")
# -------------------------------------------------
# 主函数
# -------------------------------------------------

def apply_endpoint_merge_patch(base_name, endpoints, edges):
    """
    从 ports_patch.json 中读取 endpoint_merge_patch / endpoint_coord_patch，
    将 build_connections.py 产生的“多余 terminal”合并效果真正落到最终结构里：
      - 更新代表点坐标（中心点）
      - 删除被合并掉的点
      - edges 里凡是引用 dropped_eid 的，一律替换为 merge_into 的 eid
      - 产生自环的边（eid1==eid2）会被丢弃
      - 简单去重（同一对 eid1/eid2/dir 只保留一条）
    返回： (new_endpoints, new_edges)
    """
    patch_path = os.path.join(PORTS_PATCH_DIR, f"{base_name}_ports_patch.json")
    if not os.path.exists(patch_path):
        return endpoints, edges

    try:
        with open(patch_path, "r") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] cannot load ports patch for merge: {patch_path} ({e})")
        return endpoints, edges

    merge_patch = js.get("endpoint_merge_patch", {}) or {}
    coord_patch = js.get("endpoint_coord_patch", {}) or {}

    if not merge_patch and not coord_patch:
        return endpoints, edges

    # resolve mapping (with path compression)
    m = {int(k): int(v.get("merge_into")) for k, v in merge_patch.items() if isinstance(v, dict) and v.get("merge_into") is not None}
    def resolve(eid: int) -> int:
        cur = int(eid)
        seen = set()
        while cur in m and cur not in seen:
            seen.add(cur)
            cur = m[cur]
        return cur

    # endpoints dict
    ep_by = {int(ep["eid"]): ep for ep in endpoints if "eid" in ep}

    # update representative coords
    for k, v in coord_patch.items():
        try:
            eid = int(k)
        except Exception:
            continue
        if eid not in ep_by:
            continue
        if isinstance(v, dict):
            if "x" in v and v["x"] is not None:
                ep_by[eid]["x"] = float(v["x"])
            if "y" in v and v["y"] is not None:
                ep_by[eid]["y"] = float(v["y"])

    # delete dropped endpoints
    dropped = set(int(k) for k in merge_patch.keys() if str(k).lstrip("-").isdigit())
    for d in list(dropped):
        if d in ep_by:
            del ep_by[d]

    # remap edges
    new_edges = []
    seen_keys = set()
    for e in edges:
        a = resolve(int(e["eid1"]))
        b = resolve(int(e["eid2"]))
        if a == b:
            continue
        ee = dict(e)
        ee["eid1"], ee["eid2"] = a, b
        key = (min(a, b), max(a, b), str(ee.get("dir", "")))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_edges.append(ee)

    new_endpoints = list(ep_by.values())
    # 保持输出稳定：按 eid 排序
    new_endpoints.sort(key=lambda x: int(x.get("eid", 0)))

    if merge_patch:
        print(f"  -> endpoint_merge applied: dropped={len(dropped)} kept={len(new_endpoints)}")
    if coord_patch:
        print(f"  -> endpoint_coord applied: updated={len(coord_patch)}")

    return new_endpoints, new_edges


def main():
    graph_list = sorted(glob(os.path.join(GRAPH_JSON_DIR, "*.json")))
    if not graph_list:
        print("[WARN] no graph json found in", GRAPH_JSON_DIR)
        return

    for gpath in graph_list:
        gname = os.path.splitext(os.path.basename(gpath))[0]
        # 例如 cir2_graph → cir2
        if gname.endswith("_graph"):
            base_name = gname[:-6]
        else:
            base_name = gname

        print("\n=== Processing", base_name, "===")

        # 1) 读 graph_json
        image_name, points, components, endpoints, edges = load_graph_json(gpath)

        # 2) 读 fuse_json（可能不存在）
        fuse_path = os.path.join(FUSE_JSON_DIR, f"{base_name}.json")
        texts = load_fuse_json(fuse_path)

        # ✅ 2.5) 合并新增补丁（细分类 & 端点角色）
        apply_type_refine_patch(base_name, components)
        # ✅ 先应用端点合并补丁（会改 endpoints + edges）
        endpoints, edges = apply_endpoint_merge_patch(base_name, endpoints, edges)
        # ✅ 再应用端点角色补丁（vitpose 的 port_role/port_conf 等）
        apply_ports_patch(base_name, endpoints)

        # 3) crossover 额外边
        extra_edges = build_crossover_edges(endpoints)
        print(f"  -> crossover extra edges: {len(extra_edges)}")

        all_edges = edges + extra_edges

        # 4) 计算 nets
        nets = build_nets(endpoints, all_edges)
        print(f"  -> nets: {len(nets)}")

        # 5) 读 lama 背景图，用于可视化 & width/height
        img_path = find_image(LAMA_IMG_DIR, base_name)
        if img_path is None:
            print(f"[WARN] missing lama image for {base_name}")
            img_h = img_w = None
        else:
            lama_img = cv2.imread(img_path)
            if lama_img is None:
                print(f"[WARN] cannot read lama image: {img_path}")
                img_h = img_w = None
            else:
                img_h, img_w = lama_img.shape[:2]

        texts = resolve_texts_to_component_instances(texts, components, img_w, img_h)

        # 6) 组合 final json
        final_js = {
            "image": image_name,
            "width": img_w,
            "height": img_h,
            "components": components,
            "points": points,
            "endpoints": endpoints,
            "edges": all_edges,
            "texts": texts,
            "nets": nets,
        }

        save_json = os.path.join(OUT_JSON, f"{base_name}.final.json")
        with open(save_json, "w") as f:
            json.dump(final_js, f, indent=2, ensure_ascii=False)
        print("[JSON-FINAL] Saved", save_json)

        # 7) 可视化
        if lama_img is not None:
            save_img = os.path.join(OUT_IMG, f"{base_name}_final.png")
            visualize_full(
                base_name,
                lama_img,
                points,
                components,
                endpoints,
                all_edges,
                texts,
                save_img,
            )


if __name__ == "__main__":
    main()
