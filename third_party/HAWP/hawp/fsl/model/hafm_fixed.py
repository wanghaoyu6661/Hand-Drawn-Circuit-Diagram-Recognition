# # hawp/fsl/model/hafm_fixed.py
# """
# Wrap 原始 HAFMencoder：
# - 在 _process_per_image 之前修正 ann：
#   1) 若 junctions 是 0~1，则乘回 (width, height)
#   2) 统一缩放到 cfg.DATASETS.TARGET.(WIDTH, HEIGHT)
#   3) edges_negative 为空 -> 安全置为空 (0,2) 张量
# - 然后调用“原始”的 _process_per_image，确保 targets 的键与结构保持不变
# """

# import types
# import torch
# import numpy as np
# from hawp.fsl.model import hafm as _orig  # 原始模块
# from torch.utils.data.dataloader import default_collate

# # 保存原始类与方法的引用
# _BaseHAFM = _orig.HAFMencoder
# _base_process = _orig.HAFMencoder._process_per_image

# class HAFMencoder(_BaseHAFM):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         # 从 cfg 读目标尺寸（务必与模型输出分辨率一致，通常 128x128；若你希望更高精度，可设 256x256，但需模型端同步）
#         self.target_h = int(cfg.DATASETS.TARGET.HEIGHT)
#         self.target_w = int(cfg.DATASETS.TARGET.WIDTH)

#     def _scale_ann_to_target(self, ann):
#         device = None
#         j = ann['junctions']
#         is_tensor = isinstance(j, torch.Tensor)
#         if is_tensor:
#             device = j.device
#             j_np = j.detach().cpu().numpy().astype(np.float32)
#         else:
#             j_np = np.asarray(j, dtype=np.float32)
    
#         w = float(ann.get('width', 512))
#         h = float(ann.get('height', 512))
#         if w <= 0 or h <= 0:
#             # print(f"[warn] invalid size in ann {ann.get('filename','?')} -> (512,512)")
#             w, h = 512.0, 512.0
    
#         if j_np.size > 0:
#             # 坐标判断：如果超出范围明显>1但<width，就认为是像素坐标
#             jmax = j_np.max()
#             if jmax <= 2.0:  # 真正的归一化坐标范围
#                 j_np[:, 0] *= w
#                 j_np[:, 1] *= h
    
#             # ⚙️ clamp 避免越界
#             j_np[:, 0] = np.clip(j_np[:, 0], 0, w - 1)
#             j_np[:, 1] = np.clip(j_np[:, 1], 0, h - 1)
    
#             sx = self.target_w / w
#             sy = self.target_h / h
#             j_np[:, 0] *= sx
#             j_np[:, 1] *= sy
    
#             j_np[:, 0] = np.clip(j_np[:, 0], 0, self.target_w - 1)
#             j_np[:, 1] = np.clip(j_np[:, 1], 0, self.target_h - 1)
    
#         # 更新回 ann
#         ann['junctions'] = torch.from_numpy(j_np).to(device) if is_tensor else j_np
    
#         en = ann.get('edges_negative', [])
#         if isinstance(en, torch.Tensor):
#             if en.numel() == 0:
#                 ann['edges_negative'] = torch.zeros((0, 2), dtype=torch.long, device=en.device)
#         else:
#             en_np = np.asarray(en, dtype=np.int64)
#             if en_np.size == 0:
#                 ann['edges_negative'] = np.zeros((0, 2), dtype=np.int64)
#             else:
#                 ann['edges_negative'] = en_np
    
#         ann['width'], ann['height'] = self.target_w, self.target_h
#         return ann


#     def _process_per_image(self, ann):
#         # print(f"[debug] ann keys: {list(ann.keys())}")


#         import os, numpy as np, torch

#         # —— _scale_ann_to_target 末尾 —— 
#         if os.environ.get("HAWP_DEBUG", "0") == "1":
#             j_dbg = ann['junctions'].detach().cpu().numpy() if isinstance(ann['junctions'], torch.Tensor) else np.asarray(ann['junctions'])
#             # print(f"[B] scaled-> target_wh=({ann['width']},{ann['height']}) "
#             #       f"junc: shape={j_dbg.shape}, min=({j_dbg[:,0].min():.2f},{j_dbg[:,1].min():.2f}) "
#             #       f"max=({j_dbg[:,0].max():.2f},{j_dbg[:,1].max():.2f}) "
#             #       f"type={type(ann['junctions']).__name__}")

#         # 预处理后，调用原始实现（避免破坏原 targets 的键与内部细节）
#         ann = self._scale_ann_to_target(ann)
#         return _base_process(self, ann)




# # ---- 全局计数器，避免刷屏，只对前5张图做深度打印 ----
# _DECODE_INSPECT_COUNT = {"n": 0, "max": 5}

# def _safe_to_list(x, maxlen=6):
#     """
#     将任意对象尽可能转为可JSON的浅拷贝（用于样例展示）
#     - numpy -> .tolist()
#     - torch.Tensor -> .detach().cpu().tolist()
#     - list/tuple -> 递归(截断)
#     - dict -> 仅keys列表
#     - 其它 -> repr 截断
#     """
#     import numpy as np, torch
#     try:
#         if isinstance(x, np.ndarray):
#             return x.tolist()[:maxlen]
#         if isinstance(x, torch.Tensor):
#             return x.detach().cpu().tolist()[:maxlen]
#         if isinstance(x, (list, tuple)):
#             out = []
#             for i, v in enumerate(x):
#                 if i >= maxlen:
#                     out.append("...(+{} more)".format(len(x)-maxlen))
#                     break
#                 out.append(_safe_to_list(v, maxlen=3))
#             return out
#         if isinstance(x, dict):
#             return {"__dict_keys__": list(x.keys())}
#         # 基本标量
#         if isinstance(x, (str, int, float, bool)) or x is None:
#             return x
#         s = repr(x)
#         return (s[:120] + "...") if len(s) > 120 else s
#     except Exception as e:
#         return f"<_safe_to_list error: {e!r}>"

# def _dump_structure(obj, name="decoded", depth=0, max_depth=4, prefix="  "):
#     """
#     递归打印任意Python对象的结构（类型/shape/dtype/长度…）
#     """
#     import numpy as np, torch
#     indent = prefix * depth
#     try:
#         if depth == 0:
#             print("\n[decode_stub][STRUCTURE INSPECT]")
#         if obj is None:
#             print(f"{indent}{name}: None")
#             return
#         # numpy
#         if isinstance(obj, np.ndarray):
#             print(f"{indent}{name}: ndarray shape={obj.shape}, dtype={obj.dtype}")
#             return
#         # torch
#         if isinstance(obj, torch.Tensor):
#             print(f"{indent}{name}: tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
#             return
#         # dict
#         if isinstance(obj, dict):
#             print(f"{indent}{name}: dict with {len(obj)} keys -> {list(obj.keys())}")
#             if depth >= max_depth: 
#                 return
#             for k, v in obj.items():
#                 _dump_structure(v, name=f"{name}['{k}']", depth=depth+1, max_depth=max_depth, prefix=prefix)
#             return
#         # list / tuple
#         if isinstance(obj, (list, tuple)):
#             print(f"{indent}{name}: {type(obj).__name__} len={len(obj)}")
#             if depth >= max_depth or len(obj) == 0:
#                 return
#             lim = min(5, len(obj))
#             for i in range(lim):
#                 _dump_structure(obj[i], name=f"{name}[{i}]", depth=depth+1, max_depth=max_depth, prefix=prefix)
#             if len(obj) > lim:
#                 print(f"{indent}{name}[...]: (omitted {len(obj)-lim} more)")
#             return
#         # 其它对象
#         # 尝试抓属性
#         if hasattr(obj, "__class__"):
#             tname = obj.__class__.__name__
#         else:
#             tname = type(obj).__name__
#         s = repr(obj)
#         if len(s) > 120:
#             s = s[:120] + "..."
#         print(f"{indent}{name}: {tname} -> {s}")
#         # 展开可迭代但非字符串
#         if depth < max_depth and hasattr(obj, "__dict__"):
#             print(f"{indent}{name}.__dict__ keys: {list(obj.__dict__.keys())}")
#     except Exception as e:
#         print(f"{indent}{name}: <_dump_structure error: {e!r}>")

# def _where_is_decode():
#     """
#     打印 decode_lines_from_hafm 函数来自的模块及源码文件路径
#     """
#     try:
#         mod = decode_lines_from_hafm.__module__
#         fn = getattr(decode_lines_from_hafm, "__code__", None)
#         path = fn.co_filename if fn else "(no __code__)"
#         print(f"[decode_stub] using decode_lines_from_hafm from module='{mod}', file='{path}'")
#     except Exception as e:
#         print(f"[decode_stub] cannot locate decode_lines_from_hafm: {e!r}")

# def _save_first_decoded_json(decoded, out_dir="/root/autodl-tmp/output_hawp_last/predict_results_fixed/_debug"):
#     """
#     将首个样本的 decoded 结果以JSON可读格式保存（自动处理 numpy/torch）
#     """
#     import os, json, numpy as np, torch
#     os.makedirs(out_dir, exist_ok=True)
#     def to_jsonable(x):
#         if isinstance(x, np.ndarray):
#             return x.tolist()
#         if isinstance(x, torch.Tensor):
#             return x.detach().cpu().tolist()
#         if isinstance(x, dict):
#             return {k: to_jsonable(v) for k, v in x.items()}
#         if isinstance(x, (list, tuple)):
#             return [to_jsonable(v) for v in x]
#         if isinstance(x, (str, int, float, bool)) or x is None:
#             return x
#         return repr(x)
#     try:
#         with open(os.path.join(out_dir, "decoded_first.json"), "w") as f:
#             json.dump(to_jsonable(decoded), f, indent=2)
#         print(f"[decode_stub] saved decoded structure to {out_dir}/decoded_first.json")
#     except Exception as e:
#         print(f"[decode_stub] save json failed: {e!r}")


import types
import torch
import numpy as np
from hawp.fsl.model import hafm as _orig  # 原始模块
from torch.utils.data.dataloader import default_collate

# 保存原始类与方法的引用
_BaseHAFM = _orig.HAFMencoder
_base_process = _orig.HAFMencoder._process_per_image

class HAFMencoder(_BaseHAFM):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 从 cfg 读目标尺寸（务必与模型输出分辨率一致，通常 128x128；若你希望更高精度，可设 256x256，但需模型端同步）
        self.target_h = int(cfg.DATASETS.TARGET.HEIGHT)
        self.target_w = int(cfg.DATASETS.TARGET.WIDTH)

    def _scale_ann_to_target(self, ann):
        device = None
        j = ann['junctions']
        is_tensor = isinstance(j, torch.Tensor)
        if is_tensor:
            device = j.device
            j_np = j.detach().cpu().numpy().astype(np.float32)
        else:
            j_np = np.asarray(j, dtype=np.float32)
    
        w = float(ann.get('width', 512))
        h = float(ann.get('height', 512))
        if w <= 0 or h <= 0:
            # print(f"[warn] invalid size in ann {ann.get('filename','?')} -> (512,512)")
            w, h = 512.0, 512.0
    
        if j_np.size > 0:
            # 坐标判断：如果超出范围明显>1但<width，就认为是像素坐标
            jmax = j_np.max()
            if jmax <= 2.0:  # 真正的归一化坐标范围
                j_np[:, 0] *= w
                j_np[:, 1] *= h
    
            # ⚙️ clamp 避免越界
            j_np[:, 0] = np.clip(j_np[:, 0], 0, w - 1)
            j_np[:, 1] = np.clip(j_np[:, 1], 0, h - 1)
    
            sx = self.target_w / w
            sy = self.target_h / h
            j_np[:, 0] *= sx
            j_np[:, 1] *= sy
    
            j_np[:, 0] = np.clip(j_np[:, 0], 0, self.target_w - 1)
            j_np[:, 1] = np.clip(j_np[:, 1], 0, self.target_h - 1)
    
        # 更新回 ann
        ann['junctions'] = torch.from_numpy(j_np).to(device) if is_tensor else j_np
    
        en = ann.get('edges_negative', [])
        if isinstance(en, torch.Tensor):
            if en.numel() == 0:
                ann['edges_negative'] = torch.zeros((0, 2), dtype=torch.long, device=en.device)
        else:
            en_np = np.asarray(en, dtype=np.int64)
            if en_np.size == 0:
                ann['edges_negative'] = np.zeros((0, 2), dtype=np.int64)
            else:
                ann['edges_negative'] = en_np
    
        ann['width'], ann['height'] = self.target_w, self.target_h
        return ann


    def _process_per_image(self, ann):
        # print(f"[debug] ann keys: {list(ann.keys())}")


        import os, numpy as np, torch

        # —— _scale_ann_to_target 末尾 —— 
        if os.environ.get("HAWP_DEBUG", "0") == "1":
            j_dbg = ann['junctions'].detach().cpu().numpy() if isinstance(ann['junctions'], torch.Tensor) else np.asarray(ann['junctions'])

        # 预处理后，调用原始实现（避免破坏原 targets 的键与内部细节）
        ann = self._scale_ann_to_target(ann)
        return _base_process(self, ann)

    def decode(self, md_pred, dis_pred=None, res_pred=None,
               conf_thresh=0.3, offset=6.0, max_lines=200000):
        """
        md_pred: [B,3,H,W]，用前两通道表示方向；dis/res 先不参与几何解码
        返回: [N,4] (x1,y1,x2,y2) —— 均在特征域坐标（H,W）中
        """
        import torch
        assert md_pred.ndim == 4 and md_pred.size(1) >= 2, "md_pred shape should be [B,>=2,H,W]"
        B, _, H, W = md_pred.shape
        device = md_pred.device
        lines_all = []
    
        for b in range(B):
            md = md_pred[b]              # (C,H,W) in [0,1]
            dx = md[0] * 2.0 - 1.0       # → [-1,1]
            dy = md[1] * 2.0 - 1.0
            # 方向归一化，避免长度影响
            norm = torch.sqrt(dx*dx + dy*dy) + 1e-6
            ux, uy = dx / norm, dy / norm
    
            # 强度图（用方向模长或叠加 dis_pred 作为权重）
            strength = norm
            if dis_pred is not None:
                strength = strength * dis_pred[b,0]  # 乘一个平滑权重（可选）
    
            mask = strength > conf_thresh
            ys, xs = torch.where(mask)
            if ys.numel() == 0:
                continue
    
            # 两端点（在特征域内）
            x1 = xs.float() - ux[ys, xs] * offset
            y1 = ys.float() - uy[ys, xs] * offset
            x2 = xs.float() + ux[ys, xs] * offset
            y2 = ys.float() + uy[ys, xs] * offset
    
            # 限幅
            x1 = x1.clamp(0, W - 1); x2 = x2.clamp(0, W - 1)
            y1 = y1.clamp(0, H - 1); y2 = y2.clamp(0, H - 1)
    
            lines = torch.stack([x1, y1, x2, y2], dim=-1)
            lines_all.append(lines)
    
        if not lines_all:
            return torch.zeros((0, 4), dtype=torch.float32, device=device)
    
        lines_pred = torch.cat(lines_all, dim=0)
        if lines_pred.size(0) > max_lines:
            lines_pred = lines_pred[:max_lines]
        return lines_pred






