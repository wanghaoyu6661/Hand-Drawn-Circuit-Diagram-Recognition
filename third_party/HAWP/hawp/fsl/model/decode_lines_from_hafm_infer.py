# ------------------------------------------------------------
# decode_lines_from_hafm_infer.py  — FINAL FIXED VERSION
# ------------------------------------------------------------
# - 修复了 p1/p2 未定义问题
# - 解码流程与训练 forward_train 完全一致
# - 支持 fc2 第二阶段重新评分
# - 兼容你项目的 get_junctions (N,2)
# ------------------------------------------------------------

import torch
import torch.nn.functional as F

@torch.no_grad()
def decode_lines_from_hafm_infer(
    hafm_dense,
    jloc_pred, joff_pred, dis_pred,
    *,
    conf_thresh_init=0.05,
    junc_thresh=0.04,
    j2l_radius=50.0,
    max_proposals=200000,
    seed_thresh=0.02,
    final_line_thresh=0.05,
    line_conn_probs=None
):
    """
    与 forward_train 使用的 decode_lines_from_hafm 逻辑保持一致
    （无 HAFM 密集评分，但包含两阶段 decode + fc2）
    """
    device = hafm_dense.device

    # ------------------------------------------------------------
    # 1) 通过 get_junctions 解 junctions（训练一致）
    # ------------------------------------------------------------
    from hawp.fsl.model.misc import get_junctions

    jmap = jloc_pred[0, 0]     # (Hf,Wf)
    joff = joff_pred[0]        # (2,Hf,Wf)

    # 训练路径 get_junctions 返回 (1,N,2)
    junctions = get_junctions(
        jmap.unsqueeze(0).unsqueeze(0),
        joff.unsqueeze(0)
    )[0]  # -> (N, 2)

    if junctions.shape[0] == 0:
        return [torch.zeros((0,4), device=device)], [torch.zeros((0,), device=device)]

    j_xy = junctions       # (N,2)
    N = j_xy.size(0)

    # ------------------------------------------------------------
    # 2) junction 两两组合（修复 p1/p2 调用顺序）
    # ------------------------------------------------------------
    idx_i = torch.arange(N, device=device).unsqueeze(1).repeat(1, N).reshape(-1)
    idx_j = torch.arange(N, device=device).unsqueeze(0).repeat(N, 1).reshape(-1)

    mask = idx_i < idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    if idx_i.numel() == 0:
        return [torch.zeros((0,4), device=device)], [torch.zeros((0,), device=device)]

    p1 = j_xy[idx_i]   # (M,2)
    p2 = j_xy[idx_j]   # (M,2)

    # ------------------------------------------------------------
    # 3) 长度初筛（训练同款）
    # ------------------------------------------------------------
    line_len = torch.norm(p1 - p2, dim=1)
    keep = (line_len >= j2l_radius * 0.25)

    if keep.sum() == 0:
        return [torch.zeros((0,4), device=device)], [torch.zeros((0,), device=device)]

    p1 = p1[keep]
    p2 = p2[keep]
    idx_i = idx_i[keep]
    idx_j = idx_j[keep]

    # ------------------------------------------------------------
    # 4) proposals
    # ------------------------------------------------------------
    lines_proposal = torch.stack(
        [p1[:,0], p1[:,1], p2[:,0], p2[:,1]],
        dim=1
    )  # (K,4)

    # 按你训练路径：junction score 如果没有 → 1
    j1_score = torch.ones(len(lines_proposal), device=device)
    j2_score = torch.ones(len(lines_proposal), device=device)

    # ------------------------------------------------------------
    # 5) 第一阶段（无 fc2）靠几何筛选
    # ------------------------------------------------------------
    if line_conn_probs is None:
        line_scores = j1_score * j2_score
        keep_final = line_scores >= seed_thresh
        return [lines_proposal[keep_final]], [line_scores[keep_final]]

    # ------------------------------------------------------------
    # 6) 第二阶段（使用 fc2，训练一致）
    # ------------------------------------------------------------
    conn = line_conn_probs.squeeze()  # (K,)
    conn = conn.to(device).clamp(0,1)

    if conn.shape[0] != lines_proposal.shape[0]:
        # 若对不上：训练代码中 fallback → 全 1
        conn = torch.ones(lines_proposal.shape[0], device=device)

    line_scores = conn * j1_score * j2_score
    keep_final = line_scores >= final_line_thresh

    return [lines_proposal[keep_final]], [line_scores[keep_final]]
