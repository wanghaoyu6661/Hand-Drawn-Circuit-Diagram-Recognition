#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate hand-drawn circuit parsing JSONs against GT.

It reports five circuit-level dimensions (matching the paper table style):
1) component classification
2) connectivity inference
3) text recognition + association
4) endpoint semantic recognition
5) overall success (all above satisfied)

Notes
-----
- This is an *image-level exact-match* evaluator for each dimension (one GT/pred JSON pair = one test image).
- Endpoint semantic dimension is evaluated strictly: if GT has no applicable endpoint semantics, any predicted
  `port_role` assignment is counted as incorrect for that image.
- fallback geometric matching is implemented via greedy IoU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import glob
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any


# ----------------------------
# Utility
# ----------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict, key: str, default=None):
    return d[key] if key in d else default


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # keep exact content as much as possible; only trim and collapse whitespace
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = a_area + b_area - inter
    return inter / den if den > 0 else 0.0


def center_dist(a: List[float], b: List[float]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def canonical_component_label(comp: dict) -> str:
    """
    Component classification label used for evaluation.
    Prioritizes refined class + subtype (when present) to capture fine-grained classification.
    """
    base = comp.get("cls_name_refined") or comp.get("cls_name") or f"cls_id:{comp.get('cls_id')}"
    subtype = comp.get("subtype")
    if subtype:
        return f"{base}|subtype:{subtype}"
    return str(base)


def build_component_match(gt: dict, pred: dict, prefer_id: bool = True) -> Tuple[Dict[int, int], List[str]]:
    """
    Returns mapping pred_comp_id -> gt_comp_id.
    """
    logs = []
    gt_comps = gt.get("components", [])
    pr_comps = pred.get("components", [])

    gt_by_id = {c["id"]: c for c in gt_comps if "id" in c}
    pr_by_id = {c["id"]: c for c in pr_comps if "id" in c}

    if prefer_id and len(gt_by_id) == len(gt_comps) == len(pr_by_id) == len(pr_comps):
        if set(gt_by_id.keys()) == set(pr_by_id.keys()):
            logs.append("component_match=id_identity")
            return {pid: pid for pid in pr_by_id.keys()}, logs

    # Greedy IoU fallback
    pairs = []
    for pc in pr_comps:
        pb = pc.get("bbox")
        if not isinstance(pb, list) or len(pb) != 4:
            continue
        for gc in gt_comps:
            gb = gc.get("bbox")
            if not isinstance(gb, list) or len(gb) != 4:
                continue
            iou = iou_xyxy(pb, gb)
            dist = center_dist(pb, gb)
            pairs.append((iou, -dist, pc["id"], gc["id"]))
    pairs.sort(reverse=True)

    used_p = set()
    used_g = set()
    mapping = {}
    for iou, neg_dist, pid, gid in pairs:
        if pid in used_p or gid in used_g:
            continue
        # allow zero-IoU if centers are closest and no better options remain
        mapping[pid] = gid
        used_p.add(pid)
        used_g.add(gid)

    logs.append(f"component_match=greedy_iou matched={len(mapping)}/{len(pr_comps)} pred to {len(gt_comps)} gt")
    return mapping, logs


def remap_comp_id(pred_comp_id: int, pred_to_gt_comp: Dict[int, int]):
    return pred_to_gt_comp.get(pred_comp_id, None)


def endpoint_terminal_rank_maps(data: dict, comp_id_map: Dict[int, int] | None = None) -> Tuple[Dict[int, Tuple], Dict[int, dict]]:
    """
    Build endpoint descriptor per endpoint_eid:
      descriptor = ("T", gt_comp_id_or_comp_id, side, rank_on_side)
    rank_on_side is determined within each component+side by sorted coordinate order.

    Returns:
      eid_to_desc, eid_to_endpoint
    """
    endpoints = data.get("endpoints", [])
    # optional remap for prediction side to GT component IDs
    def mapped_comp(ep):
        cid = ep.get("comp_id")
        if cid is None:
            return None
        if comp_id_map is None:
            return cid
        return comp_id_map.get(cid, None)

    per_group = defaultdict(list)  # (mapped_comp_id, side) -> [ep]
    eid_to_ep = {}
    for ep in endpoints:
        eid = ep.get("eid")
        if eid is None:
            continue
        eid_to_ep[eid] = ep
        if ep.get("kind") != "terminal":
            continue
        cid = mapped_comp(ep)
        side = ep.get("side")
        if cid is None or side is None:
            continue
        per_group[(cid, side)].append(ep)

    def sort_key(ep: dict):
        side = ep.get("side")
        # rank along the side direction to stabilize terminal identity.
        # use integer snapped coords first, then raw coords, then eid.
        x = ep.get("x", 0)
        y = ep.get("y", 0)
        x_raw = ep.get("x_raw", x)
        y_raw = ep.get("y_raw", y)
        eid = ep.get("eid", -1)
        if side in ("left", "right"):
            return (float(y), float(y_raw), float(x), float(x_raw), int(eid))
        elif side in ("top", "bottom"):
            return (float(x), float(x_raw), float(y), float(y_raw), int(eid))
        else:
            return (float(x), float(y), float(x_raw), float(y_raw), int(eid))

    eid_to_desc = {}
    for (cid, side), eps in per_group.items():
        eps_sorted = sorted(eps, key=sort_key)
        for rank, ep in enumerate(eps_sorted):
            eid_to_desc[ep["eid"]] = ("T", int(cid), str(side), int(rank))

    # node endpoints are intentionally not ranked/identified (not needed for current dimensions)
    return eid_to_desc, eid_to_ep


def component_dimension_pass(gt: dict, pred: dict, pred_to_gt_comp: Dict[int, int]) -> Tuple[bool, dict]:
    gt_comps = gt.get("components", [])
    pr_comps = pred.get("components", [])

    if len(gt_comps) != len(pr_comps):
        return False, {"reason": "component_count_mismatch", "gt": len(gt_comps), "pred": len(pr_comps)}

    gt_label = {c["id"]: canonical_component_label(c) for c in gt_comps if "id" in c}
    pr_label = {c["id"]: canonical_component_label(c) for c in pr_comps if "id" in c}

    # compare after mapping pred->gt
    mismatches = []
    mapped_gt_ids = set()
    for pid, plab in pr_label.items():
        gid = pred_to_gt_comp.get(pid)
        if gid is None:
            mismatches.append({"pred_comp_id": pid, "reason": "unmatched_component"})
            continue
        mapped_gt_ids.add(gid)
        glab = gt_label.get(gid)
        if glab != plab:
            mismatches.append({"pred_comp_id": pid, "gt_comp_id": gid, "gt": glab, "pred": plab})

    # missing gt components
    for gid in gt_label.keys():
        if gid not in mapped_gt_ids:
            mismatches.append({"gt_comp_id": gid, "reason": "missing_pred_component_match"})

    return (len(mismatches) == 0), {"mismatches": mismatches}


def text_dimension_pass(gt: dict, pred: dict, pred_to_gt_comp: Dict[int, int]) -> Tuple[bool, dict]:
    def text_sig_list(data: dict, comp_map: Dict[int, int] | None = None):
        items = []
        for t in data.get("texts", []):
            txt = normalize_text(t.get("text", ""))
            rel = t.get("match_relation", "")
            cid = t.get("matched_component_id")
            if comp_map is not None and cid is not None:
                cid = comp_map.get(cid)
            cls_name = t.get("component_cls_name") or t.get("component_name") or ""
            items.append((txt, rel, cid, cls_name))
        items.sort()
        return items

    gt_items = text_sig_list(gt, None)
    pr_items = text_sig_list(pred, pred_to_gt_comp)

    # If any pred text couldn't be mapped to component, the tuple contains None and will likely mismatch.
    return (gt_items == pr_items), {"gt_count": len(gt_items), "pred_count": len(pr_items), "gt_items": gt_items, "pred_items": pr_items}


def endpoint_semantic_dimension_pass(gt: dict, pred: dict, pred_to_gt_comp: Dict[int, int]) -> Tuple[bool, dict]:
    gt_eid_desc, gt_eid_ep = endpoint_terminal_rank_maps(gt, None)
    pr_eid_desc, pr_eid_ep = endpoint_terminal_rank_maps(pred, pred_to_gt_comp)

    def semantic_map(eid_to_desc, endpoints):
        out = {}
        for ep in endpoints:
            role = ep.get("port_role")
            if role is None:
                continue
            eid = ep.get("eid")
            desc = eid_to_desc.get(eid)
            if desc is None:
                # terminal identity unavailable -> force mismatch record
                out[("UNMAPPED_EID", eid)] = role
            else:
                out[desc] = role
        return out

    gt_map = semantic_map(gt_eid_desc, gt.get("endpoints", []))
    pr_map = semantic_map(pr_eid_desc, pred.get("endpoints", []))

    # Strict protocol: if endpoint semantics are not applicable in GT,
    # prediction must NOT assign any endpoint semantic labels.
    if len(gt_map) == 0:
        ok = (len(pr_map) == 0)
        return ok, {
            "na": True,
            "gt_semantic_count": 0,
            "pred_semantic_count": len(pr_map),
            "spurious_pred_semantic_when_na": (len(pr_map) > 0)
        }

    return (gt_map == pr_map), {
        "na": False,
        "gt_semantic_count": len(gt_map),
        "pred_semantic_count": len(pr_map),
        "gt_map": gt_map,
        "pred_map": pr_map
    }


def connectivity_dimension_pass(gt: dict, pred: dict, pred_to_gt_comp: Dict[int, int]) -> Tuple[bool, dict]:
    """
    Connectivity is evaluated from `nets` + `endpoints`, using terminal identities:
      ("T", comp_id, side, rank_on_side)
    Port semantics (`port_role`) are intentionally ignored here (evaluated separately).
    Node endpoints are ignored to focus on terminal connectivity topology.
    """
    gt_eid_desc, _ = endpoint_terminal_rank_maps(gt, None)
    pr_eid_desc, _ = endpoint_terminal_rank_maps(pred, pred_to_gt_comp)

    def net_signatures(data: dict, eid_to_desc: Dict[int, Tuple]):
        sigs = []
        for net in data.get("nets", []):
            terms = []
            for eid in net.get("endpoint_eids", []):
                desc = eid_to_desc.get(eid)
                if desc is None:
                    continue  # node endpoints or unmapped
                terms.append(desc)
            # keep multiplicity; canonical sorted tuple
            sigs.append(tuple(sorted(terms)))
        sigs.sort()
        return sigs

    gt_sigs = net_signatures(gt, gt_eid_desc)
    pr_sigs = net_signatures(pred, pr_eid_desc)

    return (gt_sigs == pr_sigs), {"gt_net_count": len(gt_sigs), "pred_net_count": len(pr_sigs), "gt_sigs": gt_sigs, "pred_sigs": pr_sigs}


def find_files(gt_dir: str, pred_dir: str, gt_suffix=".gt.json", pred_suffix=".final.json"):
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, f"*{gt_suffix}")))
    pairs = []
    missing = []
    for gp in gt_paths:
        name = os.path.basename(gp)
        stem = name[:-len(gt_suffix)]
        pp = os.path.join(pred_dir, f"{stem}{pred_suffix}")
        if os.path.exists(pp):
            pairs.append((stem, gp, pp))
        else:
            missing.append((stem, gp, pp))
    return pairs, missing


def evaluate_one(gt_path: str, pred_path: str, prefer_id_match: bool = True) -> dict:
    gt = load_json(gt_path)
    pred = load_json(pred_path)

    comp_map, match_logs = build_component_match(gt, pred, prefer_id=prefer_id_match)

    comp_ok, comp_detail = component_dimension_pass(gt, pred, comp_map)
    conn_ok, conn_detail = connectivity_dimension_pass(gt, pred, comp_map)
    text_ok, text_detail = text_dimension_pass(gt, pred, comp_map)
    sem_ok, sem_detail = endpoint_semantic_dimension_pass(gt, pred, comp_map)
    overall_ok = comp_ok and conn_ok and text_ok and sem_ok

    return {
        "image": gt.get("image") or os.path.basename(gt_path),
        "gt_path": gt_path,
        "pred_path": pred_path,
        "component_classification": bool(comp_ok),
        "connectivity_inference": bool(conn_ok),
        "text_recognition_and_association": bool(text_ok),
        "endpoint_semantic_recognition": bool(sem_ok),
        "overall_success": bool(overall_ok),
        "component_match_logs": match_logs,
        "details": {
            "component": comp_detail,
            "connectivity": conn_detail,
            "text": text_detail,
            "endpoint_semantic": sem_detail,
        }
    }


def percentage(num: int, den: int) -> float:
    return 0.0 if den == 0 else round(100.0 * num / den, 2)


def summarize(results: List[dict]) -> dict:
    n = len(results)
    k_comp = sum(r["component_classification"] for r in results)
    k_conn = sum(r["connectivity_inference"] for r in results)
    k_text = sum(r["text_recognition_and_association"] for r in results)
    k_sem = sum(r["endpoint_semantic_recognition"] for r in results)
    k_all = sum(r["overall_success"] for r in results)

    return {
        "num_circuits": n,
        "dimension_counts": {
            "component_classification": [k_comp, n],
            "connectivity_inference": [k_conn, n],
            "text_recognition_and_association": [k_text, n],
            "endpoint_semantic_recognition": [k_sem, n],
            "overall_success": [k_all, n],
        },
        "dimension_accuracy_percent": {
            "Correct component classification": percentage(k_comp, n),
            "Correct connectivity inference": percentage(k_conn, n),
            "Correct text recognition and association": percentage(k_text, n),
            "Correct endpoint semantic recognition": percentage(k_sem, n),
            "All criteria satisfied (overall success)": percentage(k_all, n),
        }
    }


def print_summary(summary: dict):
    print("=" * 72)
    print(f"Evaluated circuits: {summary['num_circuits']}")
    print("-" * 72)
    dims = summary["dimension_accuracy_percent"]
    for k, v in dims.items():
        print(f"{k:<52} {v:>6.2f}%")
    print("=" * 72)



def make_jsonable(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # convert non-JSON keys (e.g., tuple) to strings
            if isinstance(k, (str, int, float, bool)) or k is None:
                kk = k
            else:
                kk = str(k)
            out[kk] = make_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(x) for x in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate circuit parsing JSONs against GT JSONs")
    parser.add_argument("--gt-dir", required=True, help="Directory containing *.gt.json")
    parser.add_argument("--pred-dir", required=True, help="Directory containing *.final.json")
    parser.add_argument("--gt-suffix", default=".gt.json")
    parser.add_argument("--pred-suffix", default=".final.json")
    parser.add_argument("--no-prefer-id-match", action="store_true", help="Disable ID-first component matching")
    parser.add_argument("--save-json", default=None, help="Path to save detailed results JSON")
    parser.add_argument("--save-csv", default=None, help="Path to save per-circuit pass/fail CSV")
    args = parser.parse_args()

    pairs, missing = find_files(args.gt_dir, args.pred_dir, args.gt_suffix, args.pred_suffix)
    if missing:
        print("[ERROR] Missing prediction files for some GT files (strict evaluation protocol requires all test images):")
        for stem, gp, pp in missing:
            print(f"  {stem}: expected {pp}")
        raise SystemExit(f"Missing prediction files for {len(missing)} GT files. Aborting to avoid denominator mismatch.")

    if not pairs:
        raise SystemExit("No GT/pred pairs found.")

    results = []
    for stem, gt_path, pred_path in pairs:
        r = evaluate_one(gt_path, pred_path, prefer_id_match=not args.no_prefer_id_match)
        results.append(r)
        print(
            f"[{stem}] "
            f"comp={int(r['component_classification'])} "
            f"conn={int(r['connectivity_inference'])} "
            f"text={int(r['text_recognition_and_association'])} "
            f"sem={int(r['endpoint_semantic_recognition'])} "
            f"overall={int(r['overall_success'])}"
        )

    summary = summarize(results)
    print_summary(summary)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(make_jsonable({"summary": summary, "results": results}), f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.save_json}")

    if args.save_csv:
        import csv
        rows = []
        for r in results:
            rows.append({
                "image": r["image"],
                "component_classification": int(r["component_classification"]),
                "connectivity_inference": int(r["connectivity_inference"]),
                "text_recognition_and_association": int(r["text_recognition_and_association"]),
                "endpoint_semantic_recognition": int(r["endpoint_semantic_recognition"]),
                "overall_success": int(r["overall_success"]),
            })
        with open(args.save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Saved] {args.save_csv}")


if __name__ == "__main__":
    main()
