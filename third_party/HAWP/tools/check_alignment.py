# /root/autodl-tmp/HAWP/tools/check_alignment.py
import os
import cv2
import math
import json
import argparse
import numpy as np
import torch

from hawp.fsl.config import get_cfg_defaults
from hawp.fsl.data import build_train_dataset  # 你当前项目里用到的数据构建函数
from hawp.fsl.model import build_model
from hawp.fsl.model.misc import non_maximum_suppression, get_junctions

def to_int_tuple(xy):
    return tuple(int(round(xy[0])), int(round(xy[1])))

def draw_debug(img, gt_lines, gt_juncs, pred_lines_adj, pred_juncs, out_path):
    vis = img.copy()
    # GT 线（绿）
    for x1,y1,x2,y2 in gt_lines:
        cv2.line(vis, to_int_tuple((x1,y1)), to_int_tuple((x2,y2)), (0,255,0), 2)
    # 预测线（蓝）
    for x1,y1,x2,y2 in pred_lines_adj:
        cv2.line(vis, to_int_tuple((x1,y1)), to_int_tuple((x2,y2)), (255,128,0), 2)
    # GT junction（黄）
    for x,y in gt_juncs:
        cv2.circle(vis, to_int_tuple((x,y)), 3, (0,255,255), -1)
    # 预测 junction（红）
    for x,y in pred_juncs:
        cv2.circle(vis, to_int_tuple((x,y)), 3, (0,0,255), -1)
    cv2.imwrite(out_path, vis)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--num-samples', type=int, default=5)
    ap.add_argument('--seed', type=int, default=123)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # 1) 载入配置 & 数据集 & 模型（eval 模式即可）
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    torch.manual_seed(args.seed)
    dataset = build_train_dataset(cfg)  # 与你训练时一致的数据管道（会返回 metas）
    model   = build_model(cfg).cuda().eval()

    # 2) 取若干样本做检查
    report = []
    for idx in range(min(args.num_samples, len(dataset))):
        batch = [dataset[idx]]  # 单张
        images = torch.stack([b['image'] for b in batch]).cuda()  # (1,3,H,W)
        metas  = [b['meta'] for b in batch]

        # 前向：拿到特征图里 jloc/joff & 提议线（loi features只需拿shape即可）
        # 这里直接用 backbone + hafm_encoder 的中间输出；若项目封装不同可改成 model(images) 再拆
        with torch.no_grad():
            out = model.hafm_encoder(model.backbone(images))
            # out: dict，含 jloc(1,1,Hf,Wf), joff(1,2,Hf,Wf), lmap/mmap 等

        jloc_pred = out['jloc']
        joff_pred = out['joff']
        Hf, Wf = jloc_pred.shape[-2:]

        # —— 尺度信息 —— #
        meta = metas[0]
        orig_w, orig_h = meta.get('image_size', (images.shape[-1], images.shape[-2]))  # W,H
        sx, sy = float(orig_w) / float(Wf), float(orig_h) / float(Hf)

        # —— GT 提取（确保类型、形状正确） —— #
        lines_gt = meta.get('lines', None)
        junc_gt  = meta.get('junc', meta.get('junctions', None))
        if lines_gt is None or junc_gt is None:
            print(f"[{idx}] skip: no GT")
            continue
        lines_gt = torch.as_tensor(lines_gt, dtype=torch.float32)
        junc_gt  = torch.as_tensor(junc_gt,  dtype=torch.float32)

        # 基础体检 1：GT 是否在像素范围内
        def pct_out_of_bounds(lines, w, h):
            x1,y1,x2,y2 = lines[:,0],lines[:,1],lines[:,2],lines[:,3]
            bad = ((x1<0)|(x1>w-1)|(x2<0)|(x2>w-1)|(y1<0)|(y1>h-1)|(y2<0)|(y2>h-1)).float().mean().item()
            return bad
        oob = pct_out_of_bounds(lines_gt, orig_w, orig_h)
        if oob > 0.01:
            print(f"[{idx}] ❌ GT lines out-of-bounds ratio={oob:.2%} 可能是归一化坐标或 EXIF/缩放未对齐")

        # 基础体检 2：尝试“(x,y)↔(y,x)、归一化/像素”4 种假设，看哪种与可视点云最接近
        # 用“GT 端点与特征图局部峰值位置”的平均距离来打分
        def to_feat(xy):  # 原图→特征图
            xy = xy.clone()
            xy[:,0] /= sx
            xy[:,1] /= sy
            return xy

        # 预测 junction（特征图坐标 → 原图）
        jmap = non_maximum_suppression(jloc_pred[0])
        joff = joff_pred[0]
        Nj = min(int(junc_gt.shape[0])*2+2, 512)
        jp_feat, _ = get_junctions(jmap, joff, topk=Nj)  # (Nj,2) 特征图坐标
        if jp_feat is None or jp_feat.numel()==0:
            print(f"[{idx}] ❌ no predicted junctions")
            continue
        jp_img = jp_feat.clone()
        jp_img[:,0] *= sx
        jp_img[:,1] *= sy

        # 四种假设对 GT junction 测试：返回与预测 junction 点云的平均最近邻距离
        def mean_nn(px, py):
            if len(px)==0: return 1e9
            gt = torch.stack([px,py], dim=1)  # (N,2)
            # 到预测 jp_img 的 NN
            d2 = torch.cdist(gt, jp_img).min(dim=1)[0]
            return d2.mean().item()

        HYP = {}
        xg, yg = junc_gt[:,0].clone(), junc_gt[:,1].clone()
        # A: 像素(x,y)
        HYP['pix_xy'] = mean_nn(xg, yg)
        # B: 像素(y,x)
        HYP['pix_yx'] = mean_nn(yg, xg)
        # C: 归一化(x,y) -> 像素
        HYP['norm_xy'] = mean_nn(xg*orig_w, yg*orig_h)
        # D: 归一化(y,x) -> 像素
        HYP['norm_yx'] = mean_nn(yg*orig_w, xg*orig_h)

        best_hyp = min(HYP, key=HYP.get)
        print(f"[{idx}] HYP scores(px): {json.dumps({k: round(v,2) for k,v in HYP.items()})}  ==> best={best_hyp}")

        if best_hyp != 'pix_xy':
            print(f"[{idx}] ⚠️ 你的 GT junction 更像 {best_hyp}，请检查标注坐标系/维度顺序/是否归一化")

        # —— 端点吸附 + 与 GT 线距离 —— #
        # 这里用简单的“候选线”采样：从 GT 线加噪生成（为了看映射是否正确）
        # 若想完全复现训练的候选线，可改成从 model 的线提议处取 topK。
        lines_cand = lines_gt.clone()
        # 在特征图尺度模拟端点采样/回投
        def to_feat_line(L):
            out = L.clone()
            out[:,[0,2]] /= sx
            out[:,[1,3]] /= sy
            return out
        Lf = to_feat_line(lines_cand)
        # 用预测 junction 吸附（在原图空间比较）
        # 先把端点映射回原图（本例就是 L 本身），然后吸附
        cost1 = torch.cdist(lines_cand[:,:2], jp_img)  # (K,Nj)
        cost2 = torch.cdist(lines_cand[:,2:], jp_img)
        dis1, idx1 = cost1.min(dim=1)
        dis2, idx2 = cost2.min(dim=1)
        r = float(getattr(cfg.MODEL.PARSING_HEAD, 'J2L_THRESHOLD', 12.0))
        keep = (idx1!=idx2) & (dis1<r) & (dis2<r)
        lines_adj = torch.cat([jp_img[idx1], jp_img[idx2]], dim=1)

        # 与 GT 线最近邻距离（两端点交换取最小）
        def line_nn(L1, L2):
            d1 = torch.sum((L1[:,None]-L2)**2, dim=-1)
            d2 = torch.sum((L1[:,None]-L2[:,[2,3,0,1]])**2, dim=-1)
            d  = torch.min(d1,d2).min(dim=1)[0].sqrt()
            return d.mean().item()
        mean_d = line_nn(lines_adj, lines_gt)

        print(f"[{idx}] result: W,H=({orig_w},{orig_h})  Wf,Hf=({Wf},{Hf})  sx,sy=({sx:.3f},{sy:.3f})  "
              f"keep={int(keep.sum())}/{len(keep)}  mean_d(pred_adj→GT)={mean_d:.2f}px")

        # —— 导出可视化 —— #
        # 还原原图（dataset 通常已做 ToTensor/标准化，这里从 meta 取原路径或从 batch['image']反归一化）
        if 'image_path' in meta:
            img = cv2.imread(meta['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 粗略反归一化（若你的数据管道恰好是 0~255/归一化，请替换为对应反变换）
            im = images[0].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()
            img = (im*255).astype(np.uint8)

        out_png = os.path.join(args.logdir, f"align_{idx:04d}.png")
        draw_debug(
            img, 
            gt_lines=lines_gt.cpu().numpy(), 
            gt_juncs=junc_gt.cpu().numpy(), 
            pred_lines_adj=lines_adj.cpu().numpy(), 
            pred_juncs=jp_img.cpu().numpy(), 
            out_path=out_png
        )
        # 汇总
        report.append({
            "idx": idx,
            "oob_gt_ratio": round(oob, 4),
            "best_hyp": best_hyp,
            "sx": sx, "sy": sy,
            "keep": int(keep.sum()),
            "cand": int(len(keep)),
            "mean_d_adj2gt": round(mean_d, 2),
            "png": out_png
        })

    # 存个 json
    with open(os.path.join(args.logdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {len(report)} samples to {args.logdir}/report.json")

if __name__ == "__main__":
    main()
