import torch
import numpy as np
import cv2
from torch.utils.data.dataloader import default_collate
from hawp.base import _C

class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.ENCODER.DIS_TH
        self.ang_th = cfg.ENCODER.ANG_TH
        self.num_static_pos_lines = cfg.ENCODER.NUM_STATIC_POS_LINES
        self.num_static_neg_lines = cfg.ENCODER.NUM_STATIC_NEG_LINES
        self.bck_weight = cfg.ENCODER.BACKGROUND_WEIGHT

    def __call__(self, annotations):
        targets, metas = [], []
        for ann in annotations:
            t, m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        return default_collate(targets), metas

    def adjacent_matrix(self, n, edges, device):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.bool, device=device)
        if edges.size(0) > 0:
            mat[edges[:, 0], edges[:, 1]] = 1
            mat[edges[:, 1], edges[:, 0]] = 1
        return mat

    def _process_per_image(self, ann):

        import os, numpy as np
        if os.environ.get("HAWP_DEBUG", "0") == "1":
            print(f"[C1] encode_in: canvas_wh=({ann['width']},{ann['height']}) "
                  f"n_junc={len(ann['junctions'])} n_ep={len(ann.get('edges_positive',[]))} "
                  f"n_en={len(ann.get('edges_negative',[]))}")
        """
        ⚙️ 优化版:
        - Junction 用 3x3 局部高斯热点代替单像素
        - 线段 lmap 用线宽=2 的粗线绘制
        其他逻辑保持一致
        """
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']

        jmap = np.zeros((height, width), dtype=np.float32)
        joff = np.zeros((2, height, width), dtype=np.float32)

        junctions[:, 0] = junctions[:, 0].clamp(min=0, max=width - 1e-3)
        junctions[:, 1] = junctions[:, 1].clamp(min=0, max=height - 1e-3)

        junctions_np = junctions.cpu().numpy()
        xint, yint = junctions_np[:, 0].astype(np.int32), junctions_np[:, 1].astype(np.int32)
        off_x = 0.5 - (junctions_np[:, 0] - np.floor(junctions_np[:, 0]))
        off_y = 0.5 - (junctions_np[:, 1] - np.floor(junctions_np[:, 1]))

        # --- 改进点 1：局部高斯热点 (3x3) ---
        for xi, yi in zip(xint, yint):
            if 1 <= xi < width - 1 and 1 <= yi < height - 1:
                jmap[yi - 1:yi + 2, xi - 1:xi + 2] = np.maximum(
                    jmap[yi - 1:yi + 2, xi - 1:xi + 2],
                    np.array([[0.25, 0.5, 0.25],
                              [0.5, 1.0, 0.5],
                              [0.25, 0.5, 0.25]], dtype=np.float32)
                )
        joff[0, yint, xint] = off_x
        joff[1, yint, xint] = off_y

        jmap = torch.from_numpy(jmap).to(device)
        joff = torch.from_numpy(joff).to(device)

        # —— 画完 jmap 之后（jmap 是 junction heatmap）——
        if os.environ.get("HAWP_DEBUG", "0") == "1":
            jmap_sum = float(jmap.sum()) if 'jmap' in locals() else -1.0
            print(f"[C2] encode_out: jmap_sum={jmap_sum:.1f}")
        
        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']

        pos_mat = self.adjacent_matrix(junctions.size(0), edges_positive, device)
        neg_mat = self.adjacent_matrix(junctions.size(0), edges_negative, device)

        lines = torch.cat((junctions[edges_positive[:, 0]], junctions[edges_positive[:, 1]]), dim=-1)
        lines_neg = torch.cat(
            (junctions[edges_negative[:2000, 0]], junctions[edges_negative[:2000, 1]]), dim=-1)

        # --- 改进点 2：线宽 2 ---
        lmap = np.zeros((height, width), dtype=np.float32)
        lines_np = lines.cpu().numpy().astype(np.int32)
        for (x1, y1, x2, y2) in lines_np:
            cv2.line(lmap, (x1, y1), (x2, y2), 1, thickness=2)
        lmap = torch.from_numpy(lmap[None, ...]).to(device)
        # ==========================================================
        # 🧩 新增：HAFM 偏移场 GT（4 通道）
        #   每个导线像素存储到两端点的偏移 (dx1,dy1,dx2,dy2)
        # ==========================================================
        # 注意：这里用 float 精度计算，避免多次 round 误差
        lines_np_f = lines.detach().cpu().numpy().astype(np.float32)
        hafm_np = np.zeros((4, height, width), dtype=np.float32)

        for x1, y1, x2, y2 in lines_np_f:
            # 为当前线段创建一个 mask，只标出这一条线上的像素
            line_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.line(
                line_mask,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                1,
                thickness=2
            )
            ys, xs = np.where(line_mask > 0)
            if ys.size == 0:
                continue

            xs_f = xs.astype(np.float32)
            ys_f = ys.astype(np.float32)

            dx1 = x1 - xs_f
            dy1 = y1 - ys_f
            dx2 = x2 - xs_f
            dy2 = y2 - ys_f

            hafm_np[0, ys, xs] = dx1
            hafm_np[1, ys, xs] = dy1
            hafm_np[2, ys, xs] = dx2
            hafm_np[3, ys, xs] = dy2

        # 以下部分保持原逻辑不动
        center_points = (lines[:, :2] + lines[:, 2:]) / 2.0
        cmap = torch.zeros((height, width), device=device)
        xx, yy = torch.meshgrid(torch.arange(width, dtype=torch.float32, device=device),
                                torch.arange(height, dtype=torch.float32, device=device),
                                indexing='xy')
        ctl_dis = torch.min(
            (xx[..., None] - center_points[None, None, :, 0]) ** 2 +
            (yy[..., None] - center_points[None, None, :, 1]) ** 2, dim=-1)[0]
        cmask = ctl_dis <= 4.0

        cxint, cyint = center_points[:, 0].long(), center_points[:, 1].long()
        cmap[cyint, cxint] = 1

        lpos = np.random.permutation(lines.cpu().numpy())[:self.num_static_pos_lines]
        lneg = np.random.permutation(lines_neg.cpu().numpy())[:self.num_static_neg_lines]
        lpos = torch.from_numpy(lpos).to(device)
        lneg = torch.from_numpy(lneg).to(device)
        lpre = torch.cat((lpos, lneg), dim=0)
        _swap = (torch.rand(lpre.size(0)) > 0.5).to(device)
        lpre[_swap] = lpre[_swap][:, [2, 3, 0, 1]]
        lpre_label = torch.cat([
            torch.ones(lpos.size(0), device=device),
            torch.zeros(lneg.size(0), device=device)
        ])

        meta = {
            'junc': junctions,
            'junctions': junctions,
            'Lpos': pos_mat,
            'Lneg': neg_mat,
            'lpre': lpre,
            'lpre_label': lpre_label,
            'lines': lines,
        }

        dismap = torch.sqrt(lmap[0] ** 2 + lmap[0] ** 2)[None]
        def _normalize(inp):
            mag = torch.sqrt(inp[0] * inp[0] + inp[1] * inp[1])
            return inp / (mag + 1e-6)

        hafm_dis = dismap.clamp(max=self.dis_th) / self.dis_th
        mask = torch.ones_like(hafm_dis) * self.bck_weight

        target = {
            'jloc': jmap[None],
            'joff': joff,
            'cloc': cmap[None],
            'dis': hafm_dis,
            'mask': mask,
            # 新增：HAFM 偏移场 GT，形状 [4,H,W]
            'hafm': torch.from_numpy(hafm_np).to(device),
        }

        # ==========================================================
        # ⚡ HAWPv2 Electric Edition：真正的 multi-direction 方向场
        #   md: [3, H, W]
        #   - 通道 0: 方向 x 分量编码到 [0,1]（反解后 → [-1,1]）
        #   - 通道 1: 方向 y 分量编码到 [0,1]
        #   - 通道 2: 线强度(是否在导线上)，背景为 0，导线为 1
        # ==========================================================
        # 背景：方向设为 (0,0) ⇒ 编码到 (0.5, 0.5)，解码后 dx=dy≈0，强度≈0
        md_np = np.zeros((3, height, width), dtype=np.float32)
        md_np[0, :, :] = 0.5   # 背景 dx = 0
        md_np[1, :, :] = 0.5   # 背景 dy = 0
        md_np[2, :, :] = 0.0   # 背景强度 = 0

        # lines: [N,4] = (x1, y1, x2, y2) in 像素坐标（已经 clamp 到图像内）
        lines_f = lines.detach().cpu().numpy().astype(np.float32)

        for x1, y1, x2, y2 in lines_f:
            dx = x2 - x1
            dy = y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5 + 1e-6
            ux = dx / length
            uy = dy / length

            # 将 [-1,1] 的方向分量线性映射到 [0,1]
            v0 = (ux + 1.0) * 0.5
            v1 = (uy + 1.0) * 0.5

            x1_i = int(round(x1))
            y1_i = int(round(y1))
            x2_i = int(round(x2))
            y2_i = int(round(y2))

            # 在 md 的三个通道上画“带方向的粗线”（thickness 与 lmap 保持一致）
            cv2.line(md_np[0], (x1_i, y1_i), (x2_i, y2_i), float(v0), thickness=2)
            cv2.line(md_np[1], (x1_i, y1_i), (x2_i, y2_i), float(v1), thickness=2)
            cv2.line(md_np[2], (x1_i, y1_i), (x2_i, y2_i), 1.0,       thickness=2)

        # 转回 torch.Tensor 并放到正确设备
        target["md"] = torch.from_numpy(md_np).to(device)

        # 目前你的 decode() 并没有用 residual，所以先保留为 0，
        # 未来如果要做端点精修，可以在这里改成真正的残差场。
        target["res"] = torch.zeros_like(target["dis"])

        return target, meta


