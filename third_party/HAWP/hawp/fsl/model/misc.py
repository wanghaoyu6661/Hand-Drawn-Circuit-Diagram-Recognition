# hawp/fsl/model/misc.py
import torch
import torch.nn.functional as F

def non_maximum_suppression(a):
    """
    安全版 NMS：
    - 支持 2D: [H, W]
    - 支持 3D: [C, H, W]
    - 支持 4D: [N, C, H, W]
    返回张量形状与输入完全一致。
    """
    import torch.nn.functional as F
    dim = a.dim()

    if dim == 2:
        # [H, W] -> [1, 1, H, W]
        a_in = a.unsqueeze(0).unsqueeze(0)
        ap = F.max_pool2d(a_in, 3, stride=1, padding=1)
        ap = ap.squeeze(0).squeeze(0)   # 回到 [H, W]
    elif dim == 3:
        # [C, H, W] -> [1, C, H, W]
        a_in = a.unsqueeze(0)
        ap = F.max_pool2d(a_in, 3, stride=1, padding=1)
        ap = ap.squeeze(0)              # 回到 [C, H, W]
    elif dim == 4:
        # [N, C, H, W] 直接池化
        ap = F.max_pool2d(a, 3, stride=1, padding=1)
    else:
        raise ValueError(f"non_maximum_suppression expects 2D/3D/4D tensor, got {dim}D with shape={tuple(a.shape)}")

    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk=300, th=0.0, allow_fallback=True):
    """
    Robust get_junctions:
    - Auto-adjust threshold when jloc too weak
    - Support multiple input shapes: [H,W], [1,H,W], [B,1,H,W]
    - Never crash refinement due to empty detection
    """
    try:
        # ==============================
        # 统一输入维度
        # ==============================
        while jloc.dim() > 2:
            jloc = jloc.squeeze(0)
        while joff.dim() > 3:
            joff = joff.squeeze(0)

        if jloc.dim() != 2:
            raise ValueError(f"Invalid jloc shape {tuple(jloc.shape)}, expected [H,W]")
        if joff.dim() != 3 or joff.size(0) != 2:
            raise ValueError(f"Invalid joff shape {tuple(joff.shape)}, expected [2,H,W]")

        H, W = jloc.size(-2), jloc.size(-1)

        # ==============================
        # 数值安全防护
        # ==============================
        jloc = torch.nan_to_num(jloc, nan=0.0, posinf=0.0, neginf=0.0)
        joff = torch.nan_to_num(joff, nan=0.0, posinf=0.0, neginf=0.0)

        K = min(int(topk), H * W)
        flat = jloc.reshape(-1)
        scores, idx = torch.topk(flat, K, largest=True, sorted=True)
        ys = (idx // W).to(torch.int64)
        xs = (idx % W).to(torch.int64)

        dx = joff[0].reshape(-1)[idx]
        dy = joff[1].reshape(-1)[idx]

        xs_f = xs.to(jloc.dtype) + dx + 0.5
        ys_f = ys.to(jloc.dtype) + dy + 0.5

        inb = (xs_f >= 0.0) & (xs_f <= (W - 1.0)) & (ys_f >= 0.0) & (ys_f <= (H - 1.0))
        val = (scores > float(th)) & inb

        # ==============================
        # fallback: 没有任何有效点
        # ==============================
        if not val.any():
            if allow_fallback:
                fallback_K = min(4, K)
                scores, idx = torch.topk(flat, fallback_K, largest=True, sorted=True)
                ys = (idx // W).to(torch.int64)
                xs = (idx % W).to(torch.int64)
                xs_f = xs.to(jloc.dtype) + 0.5
                ys_f = ys.to(jloc.dtype) + 0.5
                junctions = torch.stack([xs_f, ys_f], dim=-1)
                print(f"[warn] get_junctions: fallback -> {fallback_K} pts (all filtered by th={th})")
                return junctions, scores
            else:
                empty_xy = jloc.new_zeros((0, 2))
                empty_sc = jloc.new_zeros((0,))
                return empty_xy, empty_sc

        xs_f = torch.clamp(xs_f[val], 0.0, W - 1.0 - 1e-3)
        ys_f = torch.clamp(ys_f[val], 0.0, H - 1.0 - 1e-3)
        scores = scores[val]
        junctions = torch.stack([xs_f, ys_f], dim=-1)

        # Debug log for sanity check
        if junctions.numel() < 2:
            print(f"[warn] get_junctions: only {junctions.size(0)} junctions kept")

        return junctions, scores

    except Exception as e:
        print(f"[fatal] get_junctions crashed: {e}")
        import traceback
        traceback.print_exc()
        # 最终兜底返回空张量，防止 refinement_train 断链
        empty_xy = jloc.new_zeros((0, 2)) if torch.is_tensor(jloc) else torch.zeros((0, 2))
        empty_sc = jloc.new_zeros((0,)) if torch.is_tensor(jloc) else torch.zeros((0,))
        return empty_xy, empty_sc


# --------------------------------------------------
# Compatibility stub for visualization (plot_lines)
# --------------------------------------------------
import numpy as np
import cv2

def plot_lines(img, lines, color=(0, 255, 0), thickness=1):
    """
    Safe stub for visualization.
    Draws lines on the input image if OpenCV available.
    """
    if img is None or lines is None:
        return img
    out = img.copy()
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
        out = (out * 255).astype(np.uint8)
        out = np.transpose(out, (1, 2, 0)) if out.ndim == 3 else out
    for l in lines:
        x1, y1, x2, y2 = map(int, l[:4])
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out
