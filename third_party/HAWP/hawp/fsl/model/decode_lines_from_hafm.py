import torch
import numpy as np
from .misc import non_maximum_suppression, get_junctions


def decode_lines_from_hafm(
    hafm_pred, jloc_pred, joff_pred, dis_pred=None,
    conf_thresh_init=0.01, junc_thresh=0.1, j2l_radius=12.0,
    max_proposals=200000, line_conn_preds=None,
    seed_thresh=None, final_line_thresh=None
):
    """
    高精度 HAFM 线段解码函数：结合 junction 得分 + 连接概率预测 + j2l 筛选。

    输入:
      - hafm_pred: HAFM 模型输出的线段张量 (shape = (C, H, W, 4)), 包含线段两端方向偏移量
      - jloc_pred: Junction heatmap 预测 (shape = (H, W) 或 (1, H, W)), 经过 sigmoid/softmax 后的概率图
      - joff_pred: Junction offset 预测 (shape = (2, H, W)), 预测的精细偏移量
      - dis_pred: (可选) 距离图预测 (shape ≈ (H, W)), 用于加强置信度计算
      - conf_thresh_init: 初始线段提案的“强度参考阈值”（主要给最终 line score 用）
      - junc_thresh: junction 提取的热图阈值 (默认 0.1)
      - j2l_radius: junction 与线段匹配半径阈值 (像素，默认 12.0)
      - max_proposals: 初始提案的最大数量上限 (默认 200000，不考虑效率时可设很大)
      - line_conn_preds: (可选) 线段存在性分类器输出的置信度 (形状 = (N_lines,)),
        由模型第二阶段分类头 (fc2) 为每个候选线段预测的存在概率。若提供，则结合该概率计算最终线段置信度；否则默认所有线段均存在。
      - seed_thresh: (可选) 像素级种子阈值，若为 None 则根据 conf_thresh_init 自适应设定
      - final_line_thresh: (可选) 最终线段置信度阈值，若为 None 则等于 conf_thresh_init

    返回:
      - lines_pred: 张量 (N, 4)，解码后的线段坐标 (x1, y1, x2, y2) (特征图坐标系)
      - line_scores: 张量 (N,) 对应每条输出线段的最终置信度
    """

    # -------------------- 1. 兼容 batch 输入 --------------------
    if hafm_pred.ndim == 5:
        # batch 模式: 对每个样本分别调用自身（递归）
        lines_batch = []
        scores_batch = []
        B = hafm_pred.shape[0]
        for b in range(B):
            pred_b = None
            if line_conn_preds is not None:
                if isinstance(line_conn_preds, (list, tuple)):
                    if len(line_conn_preds) > b:
                        pred_b = line_conn_preds[b]
                else:
                    pred_b = line_conn_preds

            lines_b, scores_b = decode_lines_from_hafm(
                hafm_pred[b],
                jloc_pred[b] if jloc_pred.ndim == 4 else jloc_pred,
                joff_pred[b] if joff_pred.ndim == 4 else joff_pred,
                dis_pred[b] if (dis_pred is not None and dis_pred.ndim == 4) else dis_pred,
                conf_thresh_init=conf_thresh_init,
                junc_thresh=junc_thresh,
                j2l_radius=j2l_radius,
                max_proposals=max_proposals,
                line_conn_preds=pred_b,
                seed_thresh=seed_thresh,
                final_line_thresh=final_line_thresh,
            )
            lines_batch.append(lines_b)
            scores_batch.append(scores_b)
        return lines_batch, scores_batch

    # -------------------- 2. 单张特征图解码 --------------------
    if hafm_pred.ndim != 4 or hafm_pred.shape[-1] != 4:
        print(f"[warn] decode_lines_from_hafm: unexpected shape {tuple(hafm_pred.shape)}, expected (C,H,W,4)")
        device = hafm_pred.device
        empty_lines = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_lines, empty_scores

    device = hafm_pred.device
    C, H, W, _ = hafm_pred.shape

    # -------------------- 3. 从 HAFM 中提取线段强度 --------------------
    strength_map = torch.linalg.norm(hafm_pred[..., :2], dim=-1)  # (C, H, W) 或 (H, W)
    if strength_map.ndim == 3:
        strength_map = strength_map.mean(dim=0)

    # dis_pred 作为额外权重
    if dis_pred is not None:
        dmap = dis_pred.squeeze().to(device, dtype=strength_map.dtype)
        if dmap.shape == strength_map.shape:
            strength_map = strength_map * dmap

    # -------------------- 3.1 seed_th / final_th 解耦 --------------------
    if seed_thresh is None:
        seed_th = max(min(float(conf_thresh_init) * 0.5, 0.05), 1e-4)
    else:
        seed_th = float(seed_thresh)

    if final_line_thresh is None:
        final_th = float(conf_thresh_init)
    else:
        final_th = float(final_line_thresh)

    # -------------------- 4. 选取候选线段中心像素 --------------------
    mask = strength_map > seed_th
    ys, xs = torch.where(mask)
    num_pts = ys.numel()

    if num_pts == 0:
        empty_lines = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_lines, empty_scores

    if num_pts > max_proposals:
        # 确保候选集合确定性（不随机）
        flat_idx = ys * W + xs
        strength_flat = strength_map.view(-1)
        seed_scores = strength_flat[flat_idx]
        topk = torch.topk(seed_scores, k=max_proposals, largest=True, sorted=False).indices
        ys = ys[topk]
        xs = xs[topk]
        num_pts = max_proposals

    # -------------------- 5. 利用 HAFM 偏移恢复线段 --------------------
    local_hafm = hafm_pred.mean(dim=0)  # (H, W, 4)
    offsets = local_hafm[ys, xs]        # (N_proposals, 4)
    dx1 = offsets[:, 0]
    dy1 = offsets[:, 1]
    dx2 = offsets[:, 2]
    dy2 = offsets[:, 3]

    x1 = xs.to(torch.float32) + dx1
    y1 = ys.to(torch.float32) + dy1
    x2 = xs.to(torch.float32) + dx2
    y2 = ys.to(torch.float32) + dy2

    x1 = torch.clamp(x1, 0, W - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    x2 = torch.clamp(x2, 0, W - 1)
    y2 = torch.clamp(y2, 0, H - 1)

    lines_pred = torch.stack([x1, y1, x2, y2], dim=-1)  # (N_proposals, 4)

    # -------------------- 6. Junction 提取 --------------------
    jloc_map = jloc_pred
    if jloc_map.dim() > 2:
        jloc_map = jloc_map.squeeze(0)  # (H, W)
    jloc_nms = non_maximum_suppression(jloc_map)

    num_candidates = int((jloc_nms > junc_thresh).float().sum().item())
    topK = min(300, num_candidates)

    joff_feat = joff_pred.squeeze(0) if joff_pred.dim() > 3 else joff_pred
    juncs_pred, juncs_score = get_junctions(jloc_nms, joff_feat, topk=topK, th=junc_thresh)

    if (juncs_pred is None) or (juncs_score is None) or (juncs_pred.numel() == 0):
        empty_lines = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_lines, empty_scores

    juncs_pred = juncs_pred.to(device, dtype=torch.float32)   # (J, 2)
    juncs_score = juncs_score.to(device, dtype=torch.float32) # (J,)

    # -------------------- 7. 线段 ↔ Junction 匹配 --------------------
    N = lines_pred.size(0)
    J = juncs_pred.size(0)

    p1 = lines_pred[:, 0:2]  # (N, 2)
    p2 = lines_pred[:, 2:4]  # (N, 2)

    p1_expand = p1.unsqueeze(1).expand(-1, J, -1)   # (N, J, 2)
    p2_expand = p2.unsqueeze(1).expand(-1, J, -1)   # (N, J, 2)
    juncs_expand = juncs_pred.unsqueeze(0).expand(N, -1, -1)  # (N, J, 2)

    dis_p1 = torch.norm(p1_expand - juncs_expand, dim=-1)  # (N, J)
    dis_p2 = torch.norm(p2_expand - juncs_expand, dim=-1)  # (N, J)

    dis1, idx1 = dis_p1.min(dim=1)  # (N,)
    dis2, idx2 = dis_p2.min(dim=1)  # (N,)

    is_valid = (dis1 <= j2l_radius) & (dis2 <= j2l_radius)
    if not bool(torch.any(is_valid)):
        empty_lines = torch.zeros((0, 4), dtype=torch.float32, device=device)
        empty_scores = torch.zeros((0,), dtype=torch.float32, device=device)
        return empty_lines, empty_scores

    idx_valid = torch.nonzero(is_valid, as_tuple=False).view(-1)
    lines_valid = lines_pred[idx_valid]
    idx1_valid = idx1[idx_valid]
    idx2_valid = idx2[idx_valid]

    juncs_p1 = juncs_pred[idx1_valid]  # (N_valid, 2)
    juncs_p2 = juncs_pred[idx2_valid]  # (N_valid, 2)

    # j2l 之后“精调过”的 proposal —— 这是【唯一一套】要给 fc2 用的线段集合
    lines_adjusted = torch.cat([juncs_p1, juncs_p2], dim=1)  # (N_valid, 4)

    # -------------------- 8. 第二阶段：若有 fc2，对同一套 proposals 打分 --------------------
    if line_conn_preds is not None:
        conn_score = line_conn_preds
        if conn_score.ndim == 2 and conn_score.size(1) == 2:
            conn_score = torch.softmax(conn_score, dim=1)[:, 1]
        else:
            conn_score = conn_score.squeeze()

        # ✅ 这里要求：fc2 的数量必须与 j2l 之后的 proposal 数量完全一致
        if conn_score.numel() != lines_adjusted.size(0):
            print(f"[warn] fc2 count {conn_score.numel()} != proposals {lines_adjusted.size(0)}, fallback to ones.")
            conn_score = torch.ones(lines_adjusted.size(0), device=device)

        # 先按 fc2 做一次过滤（存在性概率）
        keep2 = conn_score >= final_th
        if not bool(torch.any(keep2)):
            # 若全部被砍掉，保留一条分数最高的，避免彻底空
            idx_max = torch.argmax(conn_score)
            keep2 = torch.zeros_like(conn_score, dtype=torch.bool)
            keep2[idx_max] = True

        # ⚠️ 注意：所有与线段一一对应的索引都要一起过滤，保证维度一致
        lines_adjusted = lines_adjusted[keep2]
        idx1_valid = idx1_valid[keep2]
        idx2_valid = idx2_valid[keep2]
        conn_score = conn_score[keep2]

        fc2_scores = conn_score
    else:
        # 若无 fc2，则默认所有 proposal 的存在性概率为 1
        fc2_scores = torch.ones(lines_adjusted.size(0), device=device)

    # -------------------- 9. 组合 junction 置信度 & 存在概率 → 最终 line score --------------------
    junc_score_1 = juncs_score[idx1_valid]
    junc_score_2 = juncs_score[idx2_valid]

    line_scores = fc2_scores * junc_score_1 * junc_score_2

    # -------------------- 10. 最终阈值过滤 + 排序 --------------------
    if line_conn_preds is None:
        # ✅ 第一轮 decode（给 fc2 提案用）：**不再用 final_th 过滤**
        #    直接输出全部 j2l 之后的 proposals，只排序。
        keep_mask = torch.ones_like(line_scores, dtype=torch.bool)
    else:
        # ✅ 第二轮 decode（带 fc2）：这里才用 final_th 做真正的“最终线段阈值”
        conf_thresh_final = float(final_th)
        keep_mask = line_scores >= conf_thresh_final

    if not bool(torch.any(keep_mask)):
        # 极端情况下，仍然可以选择保留一条最高分线段，这里先保持为空输出，由上游处理
        lines_final = lines_adjusted.new_zeros((0, 4))
        scores_final = line_scores.new_zeros((0,))
        return lines_final, scores_final

    lines_adjusted = lines_adjusted[keep_mask]
    line_scores = line_scores[keep_mask]

    if line_scores.numel() > 0:
        order = torch.argsort(line_scores, descending=True)
        lines_adjusted = lines_adjusted[order]
        line_scores = line_scores[order]

    return lines_adjusted, line_scores
