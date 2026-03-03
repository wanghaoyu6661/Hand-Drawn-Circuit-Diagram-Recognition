import torch
from torch import nn
from hawp.fsl.backbones import build_backbone
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  numpy as np
import time
from scipy.optimize import linear_sum_assignment

from .hafm_fixed import HAFMencoder
from .losses import cross_entropy_loss_for_junction, sigmoid_l1_loss, sigmoid_focal_loss
from .misc import non_maximum_suppression, get_junctions, plot_lines
from hawp.fsl.model.decode_lines_from_hafm import decode_lines_from_hafm
from hawp.fsl.model.decode_lines_from_hafm_infer import decode_lines_from_hafm_infer


def argsort2d(arr):
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]

def nms_j(heatmap, delta=1):
    DX = [0, 0, 1, -1, 1, 1, -1, -1]
    DY = [1, -1, 0, 0, 1, -1, 1, -1]
    heatmap = heatmap.copy()
    disable = np.zeros_like(heatmap, dtype=bool)
    for x, y in argsort2d(heatmap):
        for dx, dy in zip(DX, DY):
            xp, yp = x + dx, y + dy
            if not (0 <= xp < heatmap.shape[0] and 0 <= yp < heatmap.shape[1]):
                continue
            if heatmap[x, y] >= heatmap[xp, yp]:
                disable[xp, yp] = True
    heatmap[disable] *= 0.6
    return heatmap
import numpy as np

def post_jheatmap(heatmap, offset=None, delta=1):
    """
    修复后的 Junction 解码函数：
    - 不再强制 top-K
    - 不再强制 >=1e-2 阈值（留给推理脚本处理）
    - 坐标与置信度严格对齐
    - offset 正确应用
    - 返回 (y, x, score)
    """
    H, W = heatmap.shape

    # --------------------------------
    # 1) NMS — 保留局部最大值
    # --------------------------------
    jmap = nms_j(heatmap, delta=delta)

    # --------------------------------
    # 2) 扁平化排序（置信度从高到低）
    # --------------------------------
    flat = jmap.ravel()
    idx_sorted = np.argsort(-flat)  # 从大到小排序（无截断）

    # 3) 取得坐标 (y,x)
    ys = idx_sorted // W
    xs = idx_sorted % W
    coords = np.stack([ys, xs], axis=1).astype(np.float32)

    # 4) 置信度
    scores = flat[idx_sorted]

    # --------------------------------
    # 5) 应用 offset
    # --------------------------------
    if offset is not None:
        # offset shape = (2, H, W), offset=[xoff,yoff]
        # 我们希望 coords = coords + offset
        # 所以 offset 顺序必须转换为 [yoff,xoff]
        yoff = offset[1]
        xoff = offset[0]

        # 根据每个 junction 的 (y,x) 添加偏移
        for i in range(coords.shape[0]):
            y_i = int(coords[i, 0])
            x_i = int(coords[i, 1])
            coords[i, 0] += yoff[y_i, x_i]
            coords[i, 1] += xoff[y_i, x_i]

    # --------------------------------
    # 6) +0.5 精度补偿（HAWP 官方逻辑）
    # --------------------------------
    coords += 0.5

    # --------------------------------
    # 7) 拼装 (y,x,score)
    # --------------------------------
    v0 = np.hstack([
        coords,
        scores[:, np.newaxis]
    ])

    return v0


def add_argument_with_cfg(parser, cfg, arg_name, cfg_name, help, mapping):
    
    parser.add_argument('--{}'.format(arg_name.replace('_','-')), 
        default = eval('cfg.{}'.format(cfg_name)),
        type = type(eval('cfg.{}'.format(cfg_name))),
        help = help
    )
    mapping[arg_name] = cfg_name

class WireframeDetector(nn.Module):
    def cli(self, cfg, argparser):
        cfg_mapping = {}
        sampling_parser = argparser.add_argument_group(title = 'sampling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(sampling_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('num_dyn_junctions','MODEL.PARSING_HEAD.N_DYN_JUNC', help = '[train] number of dynamic junctions')
        add_argument_lambda('num_dyn_positive_lines', 'MODEL.PARSING_HEAD.N_DYN_POSL', help ='[train] number of dynamic positive lines')
        add_argument_lambda('num_dyn_negative_lines','MODEL.PARSING_HEAD.N_DYN_NEGL', help='[train] number of dynamic negative lines')
        add_argument_lambda('num_dyn_natural_lines', 'MODEL.PARSING_HEAD.N_DYN_OTHR2', help='[train] number of dynamic line samples from the natural selection')

        matching_parser = argparser.add_argument_group(title = 'matching specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(matching_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)

        add_argument_lambda('j2l_threshold','MODEL.PARSING_HEAD.J2L_THRESHOLD', help='[all] the matching distance (in pixels^2) between the junctions and the learned lines')
        add_argument_lambda('jmatch_threshold', 'MODEL.PARSING_HEAD.JMATCH_THRESHOLD', help='[train] the matching distance (in pixels) between the predicted and grountruth junctions')

        loi_parser = argparser.add_argument_group(title = 'LOI-pooling specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(loi_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_points', 'MODEL.LOI_POOLING.NUM_POINTS', help='[train] the number of sampling points')
        add_argument_lambda('dim_junction', 'MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE', help='[train] the dim of junction features')
        add_argument_lambda('dim_edge', 'MODEL.LOI_POOLING.DIM_EDGE_FEATURE', help='[train] the dim of edge features')
        add_argument_lambda('dim_fc', 'MODEL.LOI_POOLING.DIM_FC', help='[train] the dim of fc features')

        hafm_parser = argparser.add_argument_group(title = 'Line proposal specification')
        add_argument_lambda = lambda arg_name, cfg_name, help: add_argument_with_cfg(hafm_parser, cfg, arg_name, cfg_name, help, mapping=cfg_mapping)
        add_argument_lambda('num_residuals', 'MODEL.PARSING_HEAD.USE_RESIDUAL', help='[all] the number of distance residuals')
        self.cfg_mapping = cfg_mapping
        
    def configure(self, cfg, args):
        configure_list = []
        for key, value in self.cfg_mapping.items():
            if getattr(args,key) != eval('cfg.'+value):
                configure_list.extend([value,getattr(args,key)])
        cfg.merge_from_list(configure_list)
    def __init__(self, cfg):
        super(WireframeDetector, self).__init__()
        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_backbone(cfg)
        # 由 backbone 报告特征通道数（Hourglass=cfg.MODEL.OUT_FEATURE_CHANNELS，ResNet=128）
        backbone_channels = getattr(self.backbone, "out_feature_channels", 256)
        print(f"[Backbone] type={type(self.backbone).__name__}, out_channels={backbone_channels}")
        # 目标特征图尺寸（通常 128x128），用于对齐 ResNet / Hourglass 输出
        self.target_h = int(cfg.DATASETS.TARGET.HEIGHT)
        self.target_w = int(cfg.DATASETS.TARGET.WIDTH)

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2= cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.topk_junctions = 300
        #Matcher
        self.j2l_threshold = cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD
        self.jmatch_threshold = cfg.MODEL.PARSING_HEAD.JMATCH_THRESHOLD
        self.jhm_threshold = cfg.MODEL.PARSING_HEAD.JUNCTION_HM_THRESHOLD

        # LOI POOLING
        self.n_pts0     = cfg.MODEL.LOI_POOLING.NUM_POINTS
        self.dim_junction_feature    = cfg.MODEL.LOI_POOLING.DIM_JUNCTION_FEATURE
        self.dim_edge_feature = cfg.MODEL.LOI_POOLING.DIM_EDGE_FEATURE
        self.dim_fc     = cfg.MODEL.LOI_POOLING.DIM_FC


        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.use_residual = int(cfg.MODEL.PARSING_HEAD.USE_RESIDUAL)

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None,None,:].cuda())
        
        assert cfg.MODEL.LOI_POOLING.TYPE in ['softmax', 'sigmoid']
        assert cfg.MODEL.LOI_POOLING.ACTIVATION in ['relu', 'gelu']

        self.loi_cls_type = cfg.MODEL.LOI_POOLING.TYPE
        self.loi_layer_norm = cfg.MODEL.LOI_POOLING.LAYER_NORM
        self.loi_activation = nn.ReLU if cfg.MODEL.LOI_POOLING.ACTIVATION == 'relu' else nn.GELU        

        self.fc1 = nn.Conv2d(backbone_channels, self.dim_junction_feature, 1)

        self.fc3 = nn.Conv2d(backbone_channels, self.dim_edge_feature, 1)
        self.fc4 = nn.Conv2d(backbone_channels, self.dim_edge_feature, 1)

        self.regional_head = nn.Conv2d(backbone_channels, 1, 1)
        # 新增：HAFM 偏移场分支，直接预测 (dx1,dy1,dx2,dy2)
        self.hafm_head = nn.Conv2d(backbone_channels, 4, 1)

        # ✅ fc2 只吃“沿线”的边缘特征（thin + aux），维度为 2 * dim_edge * (n_pts0-2)
        in_dim_fc2 = (self.n_pts0 - 2) * self.dim_edge_feature * 2
        fc2 = [nn.Linear(in_dim_fc2, self.dim_fc)]
        for i in range(2):
            fc2.append(self.loi_activation())
            fc2.append(nn.Linear(self.dim_fc, self.dim_fc))

        self.fc2 = nn.Sequential(*fc2)

        # 保持 fc2_res 不变，它本来也只吃 edge 特征
        self.fc2_res = nn.Sequential(
            nn.Linear(2 * (self.n_pts0 - 2) * self.dim_edge_feature, self.dim_fc),
            self.loi_activation()
        )

        self.line_mlp = nn.Sequential(
            nn.Linear((self.n_pts0-2)*self.dim_edge_feature,128),
            nn.ReLU(True),
            nn.Linear(128,32),nn.ReLU(True),
            nn.Linear(32,1)
        )

        if self.loi_cls_type == 'softmax':
            self.fc2_head = nn.Linear(self.dim_fc, 2)
            self.loss = nn.CrossEntropyLoss(reduction='none')
        elif self.loi_cls_type == 'sigmoid':
            self.fc2_head = nn.Linear(self.dim_fc, 1)
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementError()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        # ------------------------- ⚙️ 统一配置注入 -------------------------
        # 用你的 YAML 里的值覆盖 refinement 阶段默认参数
        ph_cfg = cfg.MODEL.PARSING_HEAD
        self.j2l_radius_px = float(getattr(ph_cfg, "J2L_THRESHOLD", 12.0))  # ← 半径(px)
        self.jmatch_threshold = float(getattr(ph_cfg, "JMATCH_THRESHOLD", 1.5))
        self.max_distance = float(getattr(ph_cfg, "MAX_DISTANCE", 5.0))
        self.n_dyn_junc = int(getattr(ph_cfg, "N_DYN_JUNC", 300))

        print(f"[init] ✅ J2L_THRESHOLD={self.j2l_radius_px}px | "
              f"JMATCH_THRESHOLD={self.jmatch_threshold} | "
              f"MAX_DISTANCE={self.max_distance} | "
              f"N_DYN_JUNC={self.n_dyn_junc}")
        
                # ✅ 从配置中读取安全下限（防止 valid_pairs 太少）
        self.cfg = cfg
        self.min_valid_pairs = getattr(cfg.MODEL, "MIN_VALID_PAIRS", 2)
        self.train_step = 0

    def bilinear_sampling(self, features, points):
        h,w = features.size(1), features.size(2)
        px, py = points[:,0], points[:,1]
        
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features[:, py0l, px0l] * (py1-py) * (px1 - px)+ features[:, py1l, px0l] * (py - py0) * (px1 - px)+ features[:, py0l, px1l] * (py1 - py) * (px - px0)+ features[:, py1l, px1l] * (py - py0) * (px - px0)

        return xp
    
    def get_line_points(self, lines_per_im):
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        sampled_points = U[:,:,None]*self.tspan + V[:,:,None]*(1-self.tspan) -0.5
        return sampled_points
    
    def compute_loi_features(self, features_per_image, lines_per_im):

        num_channels = features_per_image.shape[0]
        h,w = features_per_image.size(1), features_per_image.size(2)
        U,V = lines_per_im[:,:2], lines_per_im[:,2:]
        tspan = self.tspan[...,1:-1]
        sampled_points = U[:,:,None]*tspan + V[:,:,None]*(1-tspan) -0.5

        sampled_points = sampled_points.permute((0,2,1)).reshape(-1,2)
        px,py = sampled_points[:,0],sampled_points[:,1]
        px0 = px.floor().clamp(min=0, max=w-1)
        py0 = py.floor().clamp(min=0, max=h-1)
        px1 = (px0 + 1).clamp(min=0, max=w-1)
        py1 = (py0 + 1).clamp(min=0, max=h-1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        xp = features_per_image[:, py0l, px0l] * (py1-py) * (px1 - px)+ features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)+ features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)+ features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        xp = xp.reshape(features_per_image.shape[0],-1,tspan.numel()).permute(1,0,2).contiguous()

        return xp.flatten(1)
    def pooling(self, features_per_line):
        # 训练 & 验证：统一用 fc2 + fc2_head 输出标量 logit
        if self.training:
            hidden = self.fc2(features_per_line)   # [N, dim_fc]
            logits = self.fc2_head(hidden)        # [N, 1] 或 [N, 2]（softmax 情况）
            if self.loi_cls_type == 'sigmoid':
                return logits.squeeze(-1)         # [N]
            else:
                # softmax 情况下，你也可以返回 [N,2]，后面 BCE/CE 再自己处理
                return logits                     # [N,2]

        # 推理阶段：直接给出“存在性概率”
        if self.loi_cls_type == 'softmax':
            hidden = self.fc2(features_per_line)
            logits = self.fc2_head(hidden)        # [N,2]
            prob   = F.softmax(logits, dim=-1)[:, 1]
            return prob                           # [N]
        else:
            hidden = self.fc2(features_per_line)
            logits = self.fc2_head(hidden)        # [N,1]
            prob   = torch.sigmoid(logits).squeeze(-1)
            return prob                           # [N]


    def sample_points_features(self, loi_features, lines_batch):
        """
        支持 Tensor 或 list 形式的 lines_batch：
          - 如果是 list: 先 concat → 统一采样 → 按 batch 大小拆回
          - 如果是 Tensor: 直接采样
        输入:
          loi_features: [B, C, Hf, Wf] 或 [C, Hf, Wf]
          lines_batch : list(Tensor [Ni,4]) 或 Tensor [N,4]
        输出:
          若输入为 list，则返回 list，每张图一个 Tensor；
          若输入为 Tensor，则返回 Tensor。
        """

        # ------------------------------------------------------------
        # 0. 处理输入 lines_batch 是 list 的情况
        # ------------------------------------------------------------
        is_list_input = isinstance(lines_batch, (list, tuple))

        if is_list_input:
            # 统计每张图的 line 数
            line_counts = [lb.shape[0] for lb in lines_batch]

            if sum(line_counts) == 0:
                # 全空，直接返回同结构
                return [torch.zeros((0, self.dim_edge_feature * (self.n_pts0 - 2)), 
                                    device=loi_features.device)
                        for _ in lines_batch]

            # concat 成一个大 tensor
            lines_concat = torch.cat(lines_batch, dim=0)
        else:
            lines_concat = lines_batch
            line_counts = None

        # ------------------------------------------------------------
        # 1. 抽取特征 (loi_features 可能是 [B,C,H,W] 或 [C,H,W])
        # ------------------------------------------------------------
        if loi_features.ndim == 4:
            # [B,C,Hf,Wf] → 训练时
            # 当前 compute_line_conn_scores 是按 batch 内各图分别调用
            # 所以这里只会有 B=1
            loi_fea = loi_features[0]
        else:
            loi_fea = loi_features

        device = loi_fea.device

        # 没有线段，直接返回
        if lines_concat.numel() == 0:
            empty = torch.zeros((0, self.dim_edge_feature * (self.n_pts0 - 2)), device=device)
            return [empty] if is_list_input else empty

        # ------------------------------------------------------------
        # 2. 计算采样点
        # ------------------------------------------------------------
        tspan = self.tspan[..., 1:-1].to(device)  # (1,1,K-2)
        U = lines_concat[:, :2].unsqueeze(-1)
        V = lines_concat[:, 2:].unsqueeze(-1)
        sampled = U * tspan + V * (1 - tspan) - 0.5

        pts = sampled.permute(0, 2, 1).reshape(-1, 2)

        # ------------------------------------------------------------
        # 3. 双线性插值
        # ------------------------------------------------------------
        C, Hf, Wf = loi_fea.shape
        px = pts[:, 0]
        py = pts[:, 1]

        px0 = torch.floor(px).clamp(0, Wf - 1)
        py0 = torch.floor(py).clamp(0, Hf - 1)
        px1 = (px0 + 1).clamp(0, Wf - 1)
        py1 = (py0 + 1).clamp(0, Hf - 1)

        x0 = px0.long()
        y0 = py0.long()
        x1 = px1.long()
        y1 = py1.long()

        w00 = (px1 - px) * (py1 - py)
        w10 = (px - px0) * (py1 - py)
        w01 = (px1 - px) * (py - py0)
        w11 = (px - px0) * (py - py1)

        feat = (
            loi_fea[:, y0, x0] * w00 +
            loi_fea[:, y0, x1] * w10 +
            loi_fea[:, y1, x0] * w01 +
            loi_fea[:, y1, x1] * w11
        )

        # N*(K-2), C
        K2 = tspan.shape[-1]
        feat = feat.permute(1, 0).reshape(-1, C * K2)

        # ------------------------------------------------------------
        # 4. 如果输入是 list，需要拆分回去
        # ------------------------------------------------------------
        if is_list_input:
            outputs = []
            start = 0
            for cnt in line_counts:
                if cnt == 0:
                    outputs.append(torch.zeros((0, C * K2), device=device))
                else:
                    outputs.append(feat[start:start+cnt])
                start += cnt
            return outputs

        return feat

        
    def compute_line_conn_scores(self, loi_features_thin, loi_features_aux, lines_batch):
        """
        Ultra 点定位版本：为避免显存爆炸，这里不再真正计算 fc2 特征，
        而是直接返回「全 1」的存在性分数。

        这样：
        - decode_lines_from_hafm 里相当于不使用 fc2（只用几何 + junction 分数）
        - 完全绕开 sample_points_features 的大规模特征采样，避免 OOM
        - junction 定位训练不受影响
        """
        import torch

        # 取一个 device
        if isinstance(loi_features_thin, torch.Tensor):
            device = loi_features_thin.device
        elif isinstance(loi_features_thin, (list, tuple)) and len(loi_features_thin) > 0:
            device = loi_features_thin[0].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 情况 1：lines_batch 是 list，每张图一个 Tensor
        if isinstance(lines_batch, (list, tuple)):
            outputs = []
            for lb in lines_batch:
                if lb is None or lb.numel() == 0:
                    outputs.append(torch.empty(0, device=device))
                else:
                    # 返回每条线段一个「1.0」作为存在概率 / logit
                    num = lb.shape[0]
                    outputs.append(torch.ones(num, device=device))
            return outputs

        # 情况 2：lines_batch 是单个 Tensor [N,4]
        if not isinstance(lines_batch, torch.Tensor) or lines_batch.numel() == 0:
            return torch.empty(0, device=device)

        num = lines_batch.shape[0]
        return torch.ones(num, device=device)



    def forward(self, images, annotations=None, targets=None, decode=True):
        """
        通用前向接口:
          - decode=True  → 返回解码后的线段结果 (默认)
          - decode=False → 返回完整 HAFM (jmap + joff + lmap)，保持训练一致
        """
        if self.training:
            return self.forward_train(images, annotations=annotations)

        # 🚀 推理阶段 decode=False，也走完整 hafm_encoder 分支
        if not decode:
            # 使用 encoder 获取完整的 5 通道输出（jmap + joff + lmap）
            hafm_pred = self.hafm_encoder(images)
            
            # 兼容性处理：有的 encoder 返回 tuple
            if isinstance(hafm_pred, (tuple, list)):
                hafm_pred = hafm_pred[0]

            # 打印调试信息确认通道
            print(f"[forward] ⚙️ hafm_pred.shape = {tuple(hafm_pred.shape)} (应为 [B,5,H,W])")

            return hafm_pred

        # ✅ decode=True → 原始完整推理流程
        return self.forward_test(images, annotations=annotations)


    def wireframe_matcher(self, juncs_pred, lines_pred, is_train=False,return_index=False):
        cost1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1)
        cost2 = torch.sum((lines_pred[:,2:]-juncs_pred[:,None])**2,dim=-1)
        
        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)
        length = torch.sum((lines_pred[:,:2]-lines_pred[:,2:])**2,dim=-1)
 

        idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

        iskeep = idx_junc_to_end_min < idx_junc_to_end_max
        if self.j2l_threshold>0:
            iskeep *= (dis1<self.j2l_threshold)*(dis2<self.j2l_threshold)
        
        idx_lines_for_junctions = torch.stack((idx_junc_to_end_min[iskeep],idx_junc_to_end_max[iskeep]),dim=1)#.unique(dim=0)
        
        idx_lines_for_junctions, inverse = torch.unique(idx_lines_for_junctions,sorted=True,return_inverse=True,dim=0)

        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(idx_lines_for_junctions.size(0)).scatter_(0, inverse, perm)
        lines_init = lines_pred[iskeep][perm]
        if is_train:
            idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
        lines_adjusted = juncs_pred[idx_lines_for_junctions].reshape(-1,4)
        
        if return_index:
            return lines_adjusted, lines_init, perm, idx_lines_for_junctions
        else:
            return lines_adjusted, lines_init, perm
    
    def junction_detection(self, images, annotations=None):
        device = images.device

        outputs, features = self.backbone(images)
        output = outputs[0]

        # 原始预测
        jloc_pred = output[:,5:7].softmax(1)[:,1:]       # (B,1,Hf,Wf)
        joff_pred = output[:,7:9].sigmoid() - 0.5        # (B,2,Hf,Wf)

        width  = annotations[0]['width']
        height = annotations[0]['height']

        # === 高分辨率 decode 尺寸（来自 config TARGET） ===
        H_t, W_t = self.target_h, self.target_w

        # === 上采样到高分辨率 ===
        jloc_up = F.interpolate(jloc_pred, size=(H_t, W_t),
                                mode="bilinear", align_corners=False)
        joff_up = F.interpolate(joff_pred, size=(H_t, W_t),
                                mode="bilinear", align_corners=False)

        # === 高分辨率 post-processing ===
        junctions = post_jheatmap(
            jloc_up[0,0].cpu().numpy(),
            offset=joff_up[0,[1,0]].cpu().numpy()
        )

        # junctions: (N, 3) → y, x, score
        scores = junctions[:, -1]
        junc    = junctions[:, :2]  # (y, x)

        # 交换成 (x, y)
        junc_xy = junc[:, [1,0]]

        # === 映射回原图坐标 ===
        junc_xy[:,0] *= float(width)  / W_t
        junc_xy[:,1] *= float(height) / H_t

        return {
            'filename': annotations[0]['filename'],
            'juncs_pred': junc_xy,
            'juncs_score': scores,
            'width': width,
            'height': height,
        }


    def forward_test(self, images, annotations=None):
        """
        Ultra 版修复后的高分辨率 Junction 推理函数（不再错误上采样到 TARGET）
        关键点：
        - 训练使用的是 ResNetUltra → Hf,Wf ≈ input/2
        - 训练中从未使用 TARGET=320×320
        - 推理必须保持与训练一致：heatmap/offset 全在 Hf×Wf 上解码
        """

        # ---------------------------------------------------------
        # 0) 兼容：部分情况下模型没有 _eval_for_training
        # ---------------------------------------------------------
        if not hasattr(self, "_eval_for_training"):
            self._eval_for_training = False

        assert images.dim() == 4, f"[forward_test] Expect BCHW, got {images.shape}"

        device = images.device

        # ---------------------------------------------------------
        # 1) backbone feature + hafm_pred
        # ---------------------------------------------------------
        outputs, features = self.backbone(images)
        out = outputs[0]                      # (B, C, Hf, Wf)
        B, C, Hf, Wf = out.shape

        # HAFM
        hafm_dense = out[:, 0:5]              # (B,5,Hf,Wf)

        # junction heatmap + offset（在 Hf×Wf 上）
        jloc_pred = out[:, 5:7].softmax(1)[:, 1:]   # (B,1,Hf,Wf)
        joff_pred = out[:, 7:9].sigmoid() - 0.5     # (B,2,Hf,Wf)

        # dis channel
        if out.shape[1] > 9:
            dis_pred = out[:, 9:10]
        else:
            dis_pred = None

        # 原图尺寸
        H0 = annotations[0]['height']
        W0 = annotations[0]['width']
        # ---------------------------------------------------------
        # 2) 直接使用 Hf×Wf（不要上采样到 TARGET！！！）
        # ---------------------------------------------------------
        # 注意：训练所有 heatmap/offset 都是在 Hf×Wf grid 上学习。
        # post_jheatmap 必须吃 Hf×Wf 原生分辨率。

        jmap_np = jloc_pred[0, 0].detach().cpu().numpy()
        joff_np = joff_pred[0, [1, 0]].detach().cpu().numpy()

        # ---------------------------------------------------------
        # 3) 高分辨率解码（保持训练一致）
        # ---------------------------------------------------------
        junc = post_jheatmap(jmap_np, offset=joff_np)

        if junc is None or len(junc) == 0:
            return {
                "junctions_pred": torch.zeros((0, 2), device=device),
                "junctions_score": torch.zeros((0,), device=device),
                "lines_pred": torch.zeros((0, 4), device=device),
                "lines_score": torch.zeros((0,), device=device),
                "hash": None,
            }

        j_scores = junc[:, -1]
        j_xy_fm = junc[:, :2]               # (y,x) in Hf×Wf

        # (y,x) → (x,y)
        j_xy = j_xy_fm[:, [1, 0]].copy()

        # ---------------------------------------------------------
        # 4) 正确映射到原图坐标
        # ---------------------------------------------------------
        # 这里必须用 (Wf, Hf) 而不是 TARGET
        scale_x = float(W0) / float(Wf)
        scale_y = float(H0) / float(Hf)

        j_xy[:, 0] *= scale_x
        j_xy[:, 1] *= scale_y

        juncs_pred_img = torch.from_numpy(j_xy).float().to(device)
        juncs_score = torch.from_numpy(j_scores).float().to(device)

        # ---------------------------------------------------------
        # 5) 若训练 eval_for_training=True，则继续跑线段 decode
        # ---------------------------------------------------------
        lines_pred = torch.zeros((0, 4), device=device)
        lines_score = torch.zeros((0,), device=device)

        if self._eval_for_training:
            try:
                geo_lines_batch, geo_scores_batch = decode_lines_from_hafm(
                    hafm_dense.detach(),
                    jloc_pred.detach(),
                    joff_pred.detach(),
                    dis_pred.detach() if dis_pred is not None else None,
                    conf_thresh_init=self.conf_init,
                    junc_thresh=self.junc_thresh,
                    j2l_radius=self.j2l_radius_px,
                    max_proposals=self.max_prop,
                    line_conn_preds=None,
                    seed_thresh=self.conf_init * 0.5,
                    final_line_thresh=self.conf_init,
                )

                if len(geo_lines_batch) > 0:
                    lp = geo_lines_batch[0]
                    ls = geo_scores_batch[0]

                    if isinstance(lp, torch.Tensor):
                        lp = lp.cpu().numpy()
                    if isinstance(ls, torch.Tensor):
                        ls = ls.cpu().numpy()

                    if lp is not None and len(lp) > 0:
                        # 映射到原图坐标
                        lp_img = lp.copy()
                        lp_img[:, 0] *= float(W0) / float(Wf)
                        lp_img[:, 1] *= float(H0) / float(Hf)
                        lp_img[:, 2] *= float(W0) / float(Wf)
                        lp_img[:, 3] *= float(H0) / float(Hf)

                        lines_pred = torch.from_numpy(lp_img).float().to(device)
                        lines_score = torch.from_numpy(ls).float().to(device)

            except Exception as e:
                print("[warn] forward_test decode error:", e)

        # ---------------------------------------------------------
        # 6) 返回结果
        # ---------------------------------------------------------
        return {
            "junctions_pred": juncs_pred_img,
            "junctions_score": juncs_score,
            "lines_pred": lines_pred,
            "lines_score": lines_score,
            "features": features,
            "hafm_pred": hafm_dense,
            "hash": None,
        }


    def focal_loss(self,input, target, gamma=2.0):
        prob = F.softmax(input, 1) 
        ce_loss = F.cross_entropy(input, target,  reduction='none')
        p_t = prob[:,1] * target + prob[:,0] * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss
    
    def refinement_train(self, lines_batch, jloc_pred, joff_pred,
                         loi_features, loi_features_thin, loi_features_aux, metas):
        """
        改进版 refinement_train：
          ✅ 统一坐标变换逻辑（特征图↔原图）
          ✅ 支持配置匹配半径
          ✅ 打印匹配统计，帮助调参
        """
        import torch
        import torch.nn.functional as F
        from collections import defaultdict
        from .misc import non_maximum_suppression, get_junctions

        cfg = getattr(self, "cfg", None)
        device = loi_features.device
        loss_dict = defaultdict(lambda: torch.tensor(0.0, device=device))
        extra_info = defaultdict(float)

        if metas and isinstance(metas, list):
            for mi, meta in enumerate(metas[:2]):
                if isinstance(meta, dict):
                    pass

    
        # ---------- 阈值配置 ----------
        j2l_radius_px = float(getattr(self, "j2l_radius_px", 12.0))
        pos_match_radius_px = float(getattr(self, "pos_match_radius_px", 100.0))
        j2l_th_sq = j2l_radius_px ** 2
        pos_match_th_sq = pos_match_radius_px ** 2
    
        B = len(metas)
    
        for i in range(B):
            # ---------- 取本图的预测线 ----------
            if torch.is_tensor(lines_batch):
                lines_pred = lines_batch[i].reshape(-1, 4).detach()
            else:
                lines_i = lines_batch[i]
                if torch.is_tensor(lines_i):
                    lines_pred = lines_i.reshape(-1, 4).detach()
                elif len(lines_i) > 0:
                    lines_pred = torch.as_tensor(lines_i, dtype=torch.float32, device=device)
                else:
                    continue
    
            if lines_pred.numel() == 0:
                print("--------------------------------------------------")
                print("refinement_train_lines_pred.numel() == 0")
                print("--------------------------------------------------")
                continue
    
            lines_pred = lines_pred.to(device=device, dtype=torch.float32)
    
            # ---------- GT ----------
            meta = metas[i]
            junction_gt = meta['junc'] if ('junc' in meta) else meta.get('junctions', None)
            if junction_gt is None or (torch.is_tensor(junction_gt) and junction_gt.numel() == 0):
                continue
            junction_gt = torch.as_tensor(junction_gt, dtype=torch.float32, device=device)
            lines_gt = meta.get('lines', None)
            if lines_gt is None:
                continue
            lines_gt = torch.as_tensor(lines_gt, dtype=torch.float32, device=device)
    
            # ---------- 尺度 ----------
            Hf, Wf = jloc_pred[i].shape[-2:]
            orig_w, orig_h = meta.get('image_size', (512, 512))
            sx, sy = (float(orig_w) / float(Wf), float(orig_h) / float(Hf))

            if i == 0 and getattr(self, "train_step", 0) % 500 == 0:
                pass

            Hf, Wf = jloc_pred[i].shape[-2:]
            orig_w, orig_h = meta.get('image_size', (512, 512))
            
            def _range4(t):  # (N,4) -> min/max/mean 长度像素
                if t is None or t.numel() == 0:
                    return (-1, -1, -1)
                lens = torch.sqrt((t[:,2]-t[:,0])**2 + (t[:,3]-t[:,1])**2)
                return (float(lens.min()), float(lens.max()), float(lens.mean()))
            
            # 预测线(特征图域，尚未缩放)
            lp_feat = lines_pred.reshape(-1,4).detach().float().to(device)
            # GT 线（来自 meta）
            lg = lines_gt.detach().float().to(device)
            # 预测 junction（特征图域）
            jmap = non_maximum_suppression(jloc_pred[i])
            joff = joff_pred[i]
            N = int(junction_gt.shape[0])
            jp_feat, _ = get_junctions(jmap, joff, topk=min(N*2+2, getattr(self,"n_dyn_junc",512)))
            if jp_feat is None or jp_feat.numel()==0:
                # print("[A] jp_feat empty")
                continue
            jp_feat = jp_feat.float().to(device)
            
            # 侦测：坐标上限
            mx_lp = float(torch.max(lp_feat))
            mx_lg = float(torch.max(lg))
            mx_jp = float(torch.max(jp_feat))
            
            def _where(maxv, Wf, Hf, ow, oh):
                lim_feat = max(Wf, Hf)*1.2
                lim_img  = max(ow, oh)*1.2
                return "feat" if maxv <= lim_feat else ("img" if maxv <= lim_img else f"weird({maxv:.1f})")
            
            
            def _axis_hint(lines, ow, oh):
                if lines is None or lines.numel()==0:
                    return "NA"
                xs = torch.cat([lines[:,0], lines[:,2]])
                ys = torch.cat([lines[:,1], lines[:,3]])
                # 计算越界比例
                px = float((xs > ow*1.01).float().mean())  # x 超出图宽
                py = float((ys > oh*1.01).float().mean())  # y 超出图高
                return f"px>{ow}:{px:.2f} py>{oh}:{py:.2f}"

            lp = lp_feat.clone()
            jp = jp_feat.clone()


            # ---------- 预测 junction ----------
            N = int(junction_gt.shape[0])
            jmap = non_maximum_suppression(jloc_pred[i])
            joff = joff_pred[i]
            juncs_pred, _ = get_junctions(jmap, joff, topk=min(N * 2 + 2, getattr(self, "n_dyn_junc", 512)))
            if (juncs_pred is None) or (juncs_pred.numel() == 0):
                continue
            
            # ✅ 保持在特征图域，不再放大
            jp = juncs_pred.to(device=device, dtype=torch.float32).clone()
            jp[:, 0] = jp[:, 0].clamp_(0, Wf - 1)
            jp[:, 1] = jp[:, 1].clamp_(0, Hf - 1)


            # ========================= [C] GT 映射回特征图域检测 =========================
            lg_feat = lg.clone()
            lg_feat[:, [0,2]] /= (float(orig_w)/float(Wf))
            lg_feat[:, [1,3]] /= (float(orig_h)/float(Hf))
            def _inrange(lines, W, H):
                xs = torch.cat([lines[:,0], lines[:,2]])
                ys = torch.cat([lines[:,1], lines[:,3]])
                ok = ((xs>=-1) & (xs<=W+1) & (ys>=-1) & (ys<=H+1)).float().mean()
                return float(ok)
            # print(f"[C] ratio(GT mapped → feat in-range) = {_inrange(lg_feat, Wf, Hf):.2f}")



            # ---------- 匹配线-端点 ----------
            cost1 = torch.sum((lp[:, :2].unsqueeze(1) - jp.unsqueeze(0)) ** 2, dim=-1)
            cost2 = torch.sum((lp[:, 2:].unsqueeze(1) - jp.unsqueeze(0)) ** 2, dim=-1)
            dis1, idx1 = cost1.min(dim=1)
            dis2, idx2 = cost2.min(dim=1)
            
            iskeep = (idx1 != idx2) & (dis1 < j2l_th_sq) & (dis2 < j2l_th_sq)
            
            # ✅ [新增] 打印过滤前后的比例
            print(f"[check-match] img={meta.get('image_path','?').split('/')[-1]} "
                  f"| pred_lines={len(lp)} junc_pred={len(jp)} "
                  f"| pairs_all={(len(lp)*len(jp)):,} valid_pairs={iskeep.sum().item()} "
                  f"| j2l_radius={j2l_radius_px}px")
            
            # ---------- 若匹配线段过少则触发安全下限 ----------
            min_valid_pairs = getattr(self, "min_valid_pairs", 2)  # 🔧 可在 YAML 里加 MODEL.MIN_VALID_PAIRS: 2
            num_valid = int(iskeep.sum().item())
            
            if num_valid < min_valid_pairs:
                mean_dis1 = float(dis1.mean().sqrt()) if dis1.numel() else 0.0
                mean_dis2 = float(dis2.mean().sqrt()) if dis2.numel() else 0.0
                print(f"[warn-match] too few valid pairs ({num_valid}) -> mean_dis=({mean_dis1:.1f},{mean_dis2:.1f}) "
                      f"th={j2l_radius_px}px | min={min_valid_pairs}")
            
                # ✅ 若全无有效线，则随机保留少量线条参与训练，避免全部跳过
                if lines_pred.shape[0] > 0:
                    num_fallback = min(min_valid_pairs, lines_pred.shape[0])
                    perm_fallback = torch.randperm(lines_pred.shape[0], device=device)[:num_fallback]
                    iskeep = torch.zeros_like(iskeep, dtype=torch.bool)
                    iskeep[perm_fallback] = True
                else:
                    continue


    
            idx_pairs = torch.stack([idx1[iskeep], idx2[iskeep]], dim=1)
            idx_pairs = torch.sort(idx_pairs, dim=1)[0]
            idx_pairs, inverse = torch.unique(idx_pairs, sorted=True, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(idx_pairs.size(0)).scatter_(0, inverse, perm)
    
            lines_init = lp[iskeep][perm]
            lines_adjusted = torch.cat((jp[idx_pairs[:, 0]], jp[idx_pairs[:, 1]]), dim=1)
    
            # ---------- 与GT匹配 ----------
            cost_a = torch.sum((lines_adjusted[:, None] - lines_gt) ** 2, dim=-1)
            cost_b = torch.sum((lines_adjusted[:, None] - lines_gt[:, [2, 3, 0, 1]]) ** 2, dim=-1)
            cost_adj = torch.min(cost_a, cost_b)
            pos_mask_dyn = (cost_adj.min(dim=1)[0] < pos_match_th_sq)
            labels_dyn = pos_mask_dyn.float()

            # ✅ [新增] 长度一致性过滤
            len_pred = torch.sqrt((lines_adjusted[:, 2]-lines_adjusted[:, 0])**2 + 
                                  (lines_adjusted[:, 3]-lines_adjusted[:, 1])**2)
            len_gt_all = torch.sqrt((lines_gt[:, 2]-lines_gt[:, 0])**2 + 
                                    (lines_gt[:, 3]-lines_gt[:, 1])**2)
            len_gt_mean = float(len_gt_all.mean()) if len_gt_all.numel() else 1.0
            len_ratio = len_pred / (len_gt_mean + 1e-6)
            mask_len = (len_ratio > 0.2) & (len_ratio < 5.0)   # 保留长度在[0.2x, 5x]范围
            before = len(labels_dyn)
            labels_dyn = labels_dyn[mask_len]
            lines_adjusted = lines_adjusted[mask_len]
            lines_init = lines_init[mask_len]
            print(f"[check-length] keep {len(labels_dyn)}/{before} lines after length filter "
                  f"(mean len_pred={len_pred.mean():.1f}px, mean len_gt={len_gt_mean:.1f}px)")


            # ========================= [E] 方向一致性检测 =========================
            # ==== [E] 方向一致性（逐对最近匹配）====
            # 先把 lines_adjusted 和 lines_gt 放在同一域（此时都是原图域）
            # 为每条 pred 找最近 GT（考虑反向等价）
            if lines_adjusted.numel() > 0 and lg.numel() > 0:
                # 两种方向的距离
                da = torch.sum((lines_adjusted[:,None,:] - lg[None,:,:])**2, dim=-1) # (Np, Ng)
                rb = lg[:, [2,3,0,1]]
                db = torch.sum((lines_adjusted[:,None,:] - rb[None,:,:])**2, dim=-1)
                dmin, arg = torch.min(torch.min(da, db), dim=1)  # (Np,)
                # 取对应 GT（按是否反向）
                choose_b = (db.gather(1, arg.unsqueeze(1)) < da.gather(1, arg.unsqueeze(1))).squeeze(1)
                gt_matched = torch.where(
                    choose_b.unsqueeze(1),
                    lg[arg][:, [2,3,0,1]],
                    lg[arg]
                )
                dx_gt = gt_matched[:,2]-gt_matched[:,0]
                dy_gt = gt_matched[:,3]-gt_matched[:,1]
                dx_pr = lines_adjusted[:,2]-lines_adjusted[:,0]
                dy_pr = lines_adjusted[:,3]-lines_adjusted[:,1]
                cos_sim = (dx_gt*dx_pr + dy_gt*dy_pr) / (
                    torch.sqrt(dx_gt**2 + dy_gt**2 + 1e-6) * torch.sqrt(dx_pr**2 + dy_pr**2 + 1e-6)
                )

    
            # ---------- 静态线拼接 ----------
            lpre = meta.get('lpre', None)
            lpre_label = meta.get('lpre_label', None)
            if (lpre is not None) and (lpre_label is not None):
                lpre = torch.as_tensor(lpre, dtype=torch.long, device=device)
                lpre_label = torch.as_tensor(lpre_label, dtype=torch.float32, device=device)
                if lpre.numel() > 0:
                    # ✅ 过滤非法索引并强制裁剪
                    valid_mask = (lpre[:, 0] < lp.shape[0]) & (lpre[:, 1] < lp.shape[0]) & \
                                 (lpre[:, 0] >= 0) & (lpre[:, 1] >= 0)
                    lpre = lpre[valid_mask].clamp(0, lp.shape[0] - 1).long()
                    lpre_label = lpre_label[valid_mask]
            
                    # ✅ 检查索引越界
                    if (lpre[:, 0] >= lp.shape[0]).any() or (lpre[:, 1] >= lp.shape[0]).any():
                        continue
            
                    if lpre.numel() == 0:
                        # print(f"[safe-skip] lpre empty after clamp in {meta.get('image_path','?')}")
                        continue
            
                    # ✅ 生成静态线
                    lines_static = torch.cat(
                        (lp[lpre[:, 0].long(), :2], lp[lpre[:, 1].long(), :2]),
                        dim=1
                    )
            
                    lines_for_train = torch.cat((lines_adjusted, lines_static), dim=0)
                    labels_for_train = torch.cat((labels_dyn, lpre_label), dim=0)
                    lines_for_train_init = torch.cat((lines_init, lp[lpre[:, 0]]), dim=0)
                else:
                    lines_for_train, labels_for_train, lines_for_train_init = \
                        lines_adjusted, labels_dyn, lines_init
            else:
                lines_for_train, labels_for_train, lines_for_train_init = \
                    lines_adjusted, labels_dyn, lines_init


            
            if lines_for_train.numel() == 0:
                continue
    
            # ---------- 打印匹配统计 ----------
            if i == 0:
                # ---------- 基本统计 ----------
                n_gt = len(lines_gt)
                n_pred = len(lines_adjusted)
                n_pred = max(n_pred, 1)  # 防止除以 0
                pos_cnt = int(labels_dyn.sum().item()) if len(labels_dyn) > 0 else 0
                pos_ratio = labels_dyn.float().mean().item() if len(labels_dyn) > 0 else 0.0

                # 假设：
                # - keep_lines: (K, 4) 特征图坐标 (x1,y1,x2,y2)
                # - lines_gt   : (G, 4) 已经在特征图坐标（经过 HAFMencoder 的 _scale_ann_to_target）:contentReference[oaicite:5]{index=5}
                # - match_indices: 你前面通过 linear_sum_assignment 或阈值匹配得到的 "pred idx -> gt idx"
                #   （如果你当前是通过 mask 或 idx_pred/idx_gt 存下的，就用那个）

                # ========================= [F] 计算预测线 vs GT 线的几何误差 =========================
                with torch.no_grad():
                    if lines_adjusted.numel() > 0 and lg.numel() > 0:
                        # lines_adjusted: (Np, 4) - 预测线（已 snap）
                        # gt_matched    : (Np, 4) - 与预测线最近的 GT 线（已考虑方向）
                        dx1 = lines_adjusted[:, 0] - gt_matched[:, 0]
                        dy1 = lines_adjusted[:, 1] - gt_matched[:, 1]
                        dx2 = lines_adjusted[:, 2] - gt_matched[:, 2]
                        dy2 = lines_adjusted[:, 3] - gt_matched[:, 3]

                        d1 = torch.sqrt(dx1 * dx1 + dy1 * dy1)
                        d2 = torch.sqrt(dx2 * dx2 + dy2 * dy2)

                        mean_d_gt = (d1.mean().item(), d2.mean().item())
                    else:
                        mean_d_gt = (0.0, 0.0)

                # ---------- 主日志 ----------
                print(f"[refine] lines_pred={n_pred}, juncs={len(jp)}, valid_pairs={iskeep.sum().item()}, "
                      f"GT_lines={n_gt}, pos={pos_cnt}, "
                      f"mean_d_gt=({mean_d_gt[0]:.2f},{mean_d_gt[1]:.2f})px, "
                      f"r={j2l_radius_px}px, pos_th={pos_match_radius_px}px")
            
                # ---------- 安全打印（带零保护） ----------
                try:
                    ratio_gt_pred = n_gt / n_pred if n_pred > 0 else 0.0
                except Exception:
                    ratio_gt_pred = 0.0
            
                print(f"         ⮑ pos_ratio={pos_ratio:.3f} ({pos_cnt}/{len(labels_dyn)}) "
                      f"| GT vs Pred ratio={n_gt}/{n_pred}={ratio_gt_pred:.2f}")
            
                # ---------- 若本图无有效样本，额外提示 ----------
                if len(labels_dyn) == 0 or n_pred == 0:
                    print(f"[warn-skip] ⚠️ no valid dynamic labels for this image "
                          f"(keep={len(labels_dyn)}, lines_pred={n_pred}), skipping safely.")


    
            # ---------- 特征图坐标采样 ----------
            def to_feat_map_xy(pts_xy):
                out = pts_xy.clone()
                out[:, 0] = out[:, 0] / sx
                out[:, 1] = out[:, 1] / sy
                return out
    
            def _bilinear_sampling(feat, pts_xy_feat):
                h, w = feat.size(1), feat.size(2)
                px, py = pts_xy_feat[:, 0], pts_xy_feat[:, 1]
                px0, py0 = px.floor().clamp(0, w - 1), py.floor().clamp(0, h - 1)
                px1, py1 = (px0 + 1).clamp(0, w - 1), (py0 + 1).clamp(0, h - 1)
                px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
                return (
                    feat[:, py0l, px0l] * (py1 - py) * (px1 - px)
                    + feat[:, py1l, px0l] * (py - py0) * (px1 - px)
                    + feat[:, py0l, px1l] * (py1 - py) * (px - px0)
                    + feat[:, py1l, px1l] * (py - py0) * (px - px0)
                )
    
            p1_fm = to_feat_map_xy(lines_for_train[:, :2]) - 0.5
            p2_fm = to_feat_map_xy(lines_for_train[:, 2:]) - 0.5
            e1 = _bilinear_sampling(loi_features[i], p1_fm).t()
            e2 = _bilinear_sampling(loi_features[i], p2_fm).t()
            lines_for_train_fm = torch.cat((p1_fm, p2_fm), dim=1)
            lines_for_train_init_fm = torch.cat(
                (to_feat_map_xy(lines_for_train_init[:, :2]) - 0.5,
                 to_feat_map_xy(lines_for_train_init[:, 2:]) - 0.5),
                dim=1
            )
    
            f1 = self.compute_loi_features(loi_features_thin[i], lines_for_train_fm)
            f2 = self.compute_loi_features(loi_features_aux[i], lines_for_train_init_fm)
            # ✅ 仅保留“沿线 edge 特征（thin + aux）”作为 fc2 的输入
            line_feat = torch.cat((f1, f2), dim=-1)   # shape = 2*(n_pts0-2)*dim_edge_feature = 240

            logits = self.fc2_head(
                self.fc2(line_feat) +                   # [N, dim_fc]
                self.fc2_res(torch.cat((f1, f2), dim=-1))
            )
    
            # ---------- 分类损失（修复 & 对称防护版） ----------
            if self.loi_cls_type == 'sigmoid':
                # BCE: logits 是 [N,1] or [N], labels_for_train 是 [N]
                bce = self.bce_loss(logits.flatten(), labels_for_train.float())

                pos_mask = (labels_for_train == 1)
                neg_mask = (labels_for_train == 0)

                # ⚠️ 正负样本都做对称防护，避免空 tensor.mean() → nan
                loss_pos = bce[pos_mask].mean() if pos_mask.any() else torch.zeros((), device=device)
                loss_neg = bce[neg_mask].mean() if neg_mask.any() else torch.zeros((), device=device)

            else:
                # CrossEntropyLoss: logits [N,2], labels [N]
                loss_all = self.loss(logits, labels_for_train.long())

                pos_mask = (labels_for_train == 1)
                neg_mask = (labels_for_train == 0)

                # ⚠️ 同样对称防护
                loss_pos = loss_all[pos_mask].mean() if pos_mask.any() else torch.zeros((), device=device)
                loss_neg = loss_all[neg_mask].mean() if neg_mask.any() else torch.zeros((), device=device)

            # ---------- 可选 Debug（默认注释掉） ----------
            num_pos = int(pos_mask.sum())
            num_neg = int(neg_mask.sum())
            # print(f"[check-loss] pos={num_pos} neg={num_neg} | loss_pos={loss_pos.item():.4f} loss_neg={loss_neg.item():.4f}")

            # 累积 batch 内所有样本
            loss_dict["loss_pos"] += loss_pos
            loss_dict["loss_neg"] += loss_neg

            # lineness 保持你的实现
            loss_dict["loss_lineness"] += torch.tensor(0.0, device=device)

            # ---------- 调试：方向相似性 & 线长（仅打印第一个样本防止刷屏） ----------
            if i == 0:
                # ============ [安全防护 v3 · 全CPU计算，彻底绕开CUDA断言] ============
                # 0) 前置校验
                if (not torch.is_tensor(cost_adj)) or (cost_adj.numel() == 0):
                    print("[safe-skip] cost_adj invalid or empty -> skip orientation stats")
                    continue
                if cost_adj.dim() != 2:
                    print(f"[safe-skip] cost_adj has invalid dim={cost_adj.dim()} -> skip orientation stats")
                    continue
                if lines_gt.numel() == 0:
                    print("[safe-skip] lines_gt empty -> skip orientation stats")
                    continue
            
                # 1) 统一搬到 CPU 做 argmin & 索引，避免 device assert
                cost_adj_cpu = cost_adj.detach().cpu()
                lines_gt_cpu = lines_gt.detach().cpu()
                lines_adjusted_cpu = lines_adjusted.detach().cpu()
                pos_mask_dyn_cpu = pos_mask_dyn.detach().cpu()
            
                # 2) 计算 idx_best_all（每条 pred 对应最近 GT 的索引）
                idx_best_all = torch.argmin(cost_adj_cpu, dim=1)  # (N_pred,)
                if idx_best_all.numel() == 0 or torch.any(torch.isnan(idx_best_all)):
                    print("[safe-skip] idx_best_all invalid/empty -> skip orientation stats")
                    continue
            
                max_idx = lines_gt_cpu.shape[0] - 1
                if max_idx < 0:
                    print("[safe-skip] no GT -> skip orientation stats")
                    continue
            
                # 3) clamp 到合法范围，并转 long
                idx_best_all = idx_best_all.clamp_(0, max_idx).long()
            
                # 4) 取正样本索引；如果没有正样本，就用全部
                valid_idx = torch.nonzero(pos_mask_dyn_cpu, as_tuple=False).squeeze(1)
                
                # ✅ 空检查：若无正样本或 lines_adjusted 为空，则安全跳过
                if (not torch.is_tensor(lines_adjusted_cpu)) or (lines_adjusted_cpu.numel() == 0):
                    print("[safe-skip] lines_adjusted_cpu empty -> skip orientation stats")
                    continue
                
                if valid_idx.numel() == 0:
                    print("[safe-skip] valid_idx empty -> skip orientation stats")
                    idx_best = idx_best_all
                    la_cpu = lines_adjusted_cpu
                else:
                    valid_idx = valid_idx.clamp(0, max(lines_adjusted_cpu.shape[0] - 1, 0))
                    if valid_idx.numel() == 0:
                        print("[safe-skip] valid_idx zero after clamp -> skip orientation stats")
                        continue
                    if valid_idx.max().item() >= lines_adjusted_cpu.shape[0]:
                        print(f"[warn-skip] valid_idx out of range: max={valid_idx.max().item()} >= lines_adjusted_cpu.shape[0]={lines_adjusted_cpu.shape[0]}")
                        continue
                    la_cpu = lines_adjusted_cpu[valid_idx]
                    idx_best = idx_best_all[valid_idx]


                # 5) 再防护：把任何越界索引过滤掉，并同步截断 la_cpu
                keep = (idx_best >= 0) & (idx_best <= max_idx)
                if keep.numel() == 0 or keep.sum() == 0:
                    print("[safe-skip] idx_best all out-of-range after clamp -> skip")
                    continue
                idx_best = idx_best[keep]
                la_cpu = la_cpu[:len(idx_best)]  # 保持一一对应
            
                if idx_best.numel() == 0:
                    print("[safe-skip] idx_best empty after keep-filter -> skip")
                    continue
            
                # 6) 最终安全取值（仍在 CPU 上）
                try:
                    gt_sel_cpu = lines_gt_cpu[idx_best].clone()  # (N,4)
                except Exception as e:
                    print(f"[safe-catch] gt_sel index error on CPU: idx_best max={int(idx_best.max()) if idx_best.numel()>0 else -1}, "
                          f"gt_lines={lines_gt_cpu.shape[0]} | err={str(e)}")
                    continue
            
                # ---------- [7] 方向与长度统计（CPU，长度同步保护） ----------
                # 若数量不一致，取两者最小长度
                n_gt = gt_sel_cpu.shape[0]
                n_pr = la_cpu.shape[0]
                n_min = min(n_gt, n_pr)
                if n_min == 0:
                    print(f"[safe-skip] no valid pairs for orientation (n_gt={n_gt}, n_pr={n_pr}) -> skip")
                    continue
                
                if n_gt != n_pr:
                    print(f"[safe-align] ⚠️ mismatch gt/pr count -> gt={n_gt}, pred={n_pr}, align to {n_min}")
                    gt_sel_cpu = gt_sel_cpu[:n_min]
                    la_cpu = la_cpu[:n_min]
                
                dx_gt = gt_sel_cpu[:, 2] - gt_sel_cpu[:, 0]
                dy_gt = gt_sel_cpu[:, 3] - gt_sel_cpu[:, 1]
                dx_pr = la_cpu[:, 2] - la_cpu[:, 0]
                dy_pr = la_cpu[:, 3] - la_cpu[:, 1]
                
                denom = (torch.sqrt(dx_gt**2 + dy_gt**2 + 1e-6) *
                         torch.sqrt(dx_pr**2 + dy_pr**2 + 1e-6))

            
                # [debug-flip] 计算正向与反向的余弦相似度
                denom = (torch.sqrt(dx_gt**2 + dy_gt**2 + 1e-6) *
                         torch.sqrt(dx_pr**2 + dy_pr**2 + 1e-6))
                cos_ori = ((dx_gt * dx_pr + dy_gt * dy_pr) / denom).clamp(-1.0, 1.0)
                cos_flip = ((dx_gt * (-dx_pr) + dy_gt * (-dy_pr)) / denom).clamp(-1.0, 1.0)
            
                mean_ori = cos_ori.mean().item() if cos_ori.numel() > 0 else 0.0
                mean_flip = cos_flip.mean().item() if cos_flip.numel() > 0 else 0.0
                # print(f"[debug-flip] ⟨cos_ori⟩={mean_ori:.3f}, ⟨cos_flip⟩={mean_flip:.3f}, Δ={mean_flip - mean_ori:+.3f}")
            
                # [debug-dir] 常规方向/长度统计（CPU）
                cos_sim = ((dx_gt * dx_pr + dy_gt * dy_pr) / denom).clamp(-1.0, 1.0)
                len_gt = torch.sqrt(dx_gt**2 + dy_gt**2)
                len_pr = torch.sqrt(dx_pr**2 + dy_pr**2)
            
                n_items = len_pr.numel()
                if n_items > 0:
                    pass
            
        return loss_dict, extra_info


    def forward_train(self, images, annotations=None):
        import torch.nn.functional as F
        device = images.device
        targets, metas = self.hafm_encoder(annotations)
        self.train_step += 1

        # =====================================================
        # 🔧 兼容新版 HAFM 输出字段 (2023+ 版本)
        # =====================================================
        if 'jloc' in targets and 'md' not in targets:
            targets['md'] = targets['jloc']
        if 'cloc' in targets and 'dis' not in targets:
            targets['dis'] = targets['cloc']
        if 'joff' in targets and 'res' not in targets:
            targets['res'] = targets['joff']

        # --- 修正通道数匹配 ---
        if 'res' in targets:
            joff_tgt = targets['res']
            if joff_tgt.ndim == 3:
                joff_tgt = joff_tgt.unsqueeze(1)
            if joff_tgt.shape[1] > 2:
                joff_tgt = joff_tgt[:, :2]
            elif joff_tgt.shape[1] < 2:
                joff_tgt = joff_tgt.repeat(1, 2, 1, 1)
            targets['res'] = joff_tgt

        outputs, features = self.backbone(images)

        import os, torch

        # —— 在各项 loss 计算之前 —— 
        if os.environ.get("HAWP_DEBUG", "0") == "1":
            output = outputs[0]
            out = output[:, :3]                # md 分支通道
            tgt = targets['md']
            # print(f"[D] out_md={tuple(out.shape)} tgt_md={tuple(tgt.shape)} "
            #       f"out_nan={~torch.isfinite(out).all().item()} tgt_nan={~torch.isfinite(tgt).all().item()}")


        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
            'loss_aux': 0.0,
            'loss_lineness': 0.0,
            'loss_hafm': 0.0,  #  监督 HAFM 偏移场
        }

        extra_info = defaultdict(list)

        # ------------------ 计算特征 ------------------
        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)

        output = outputs[0]
        md_pred  = output[:, :3].sigmoid()
        dis_pred = output[:, 3:4].sigmoid()
        res_pred = output[:, 4:5].sigmoid()
        jloc_pred = output[:, 5:7].softmax(1)[:, 1:]
        joff_pred = output[:, 7:9].sigmoid() - 0.5

        # HAFM 偏移场预测：直接从 backbone feature 上回归 4 通道
        hafm_pred_map = self.hafm_head(features)  # [B,4,Hf,Wf]

        # 🔧 若 ResNet 输出为 64x64，则上采样到 128x128 对齐 HAFMencoder 目标
        if hafm_pred_map.shape[-2] != self.target_h or hafm_pred_map.shape[-1] != self.target_w:
            import torch.nn.functional as F
            hafm_pred_map = F.interpolate(
                hafm_pred_map,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            )

        batch_size = md_pred.size(0)

        # ------------------ HAFM 解码（直接使用偏移场预测） ------------------
        # 我们把 [B,4,H,W] 变成 [B,1,H,W,4]，让 decode_lines_from_hafm 复用原有逻辑
        B, _, Hf, Wf = hafm_pred_map.shape
        hafm_dense = hafm_pred_map.permute(0, 2, 3, 1).unsqueeze(1)  # [B,1,H,W,4]

        # 此时 hafm_dense 形状一般为 [B, C, H, W, 4]

        # ============================================================
        # ✅ 新版：dense → coordinate 解码（双动态阈值 + 指数 warmup）
        # ============================================================

        # ---------- 1) 超长 warmup 的基础设置 ----------
        # 说明：
        #  - 线段强度在训练早期通常很弱，如果阈值上升太快，会导致 decode 直接“死掉”
        #  - 因此使用较长 warmup，并且用指数曲线减缓前期上升速度
        long_warmup_steps = 50000.0  # 超长 warmup（可根据总 iter 适当调整）

        # 当前训练进度比例（0~1）
        lin_ratio = min(1.0, float(self.train_step) / long_warmup_steps)
        # 指数型 warmup：前期极慢，后期再加快
        ratio = lin_ratio ** 2.5

        # ---------- 2) 动态线段强度阈值 conf_th ----------
        # 起始值尽量低，保证 early recall；上限也不要太高
        conf_base = 0.01   # 初始强度阈值
        conf_max  = 0.10   # 最终强度阈值上限（不建议再高）
        conf_th = conf_base + (conf_max - conf_base) * ratio

        # ---------- 3) 动态 junction 热图阈值 junc_thresh ----------
        # 同样采用低起点 + 指数 warmup
        junc_base = 0.01   # 初始 junc 阈值
        junc_max  = 0.06   # 最终 junc 阈值上限
        junc_thresh = junc_base + (junc_max - junc_base) * ratio

        # ---------- 4) J2L 半径：与 train.py 中动态调度保持一致 ----------
        # train.py 中每个 epoch 会写入 model.j2l_radius_px
        j2l_radius = float(getattr(self, "j2l_radius_px", 60.0))

        # ---------- 5) 高精度 decode（第一轮尝试） ----------
        # hafm_dense: [B,C,H,W,4]
        # jloc_pred:  [B,1,H,W]
        # joff_pred:  [B,2,H,W]
        # dis_pred:   [B,1,H,W]
        # ======= 新 decode：需要加入 line_conn_preds（第二阶段分类器输出） =======
        # ============================================================
        # （假设前面你已经算好 conf_th / junc_thresh / seed_th）
        # ============================================================

        # 1) 先用「纯几何」 decode 得到候选线段
        geo_lines_batch, geo_scores_batch = decode_lines_from_hafm(
            hafm_dense.detach(),
            jloc_pred.detach(),
            joff_pred.detach(),
            dis_pred.detach(),

            conf_thresh_init=conf_th,
            junc_thresh=junc_thresh,
            j2l_radius=self.j2l_radius_px,
            max_proposals=10000,

            line_conn_preds=None,      # 第一阶段：不用 fc2
            seed_thresh=conf_th * 0.5,
            final_line_thresh=conf_th,
        )

        # 2) 用 fc2 对候选线段打存在性概率（训练阶段也要用，保持 train/test 一致）
        line_conn_preds = self.compute_line_conn_scores(
            loi_features_thin,
            loi_features_aux,
            geo_lines_batch,
        )

        # 3) 带 fc2 的最终 decode，输出 lines_batch / line_scores_batch
        lines_batch, line_scores_batch = decode_lines_from_hafm(
            hafm_dense.detach(),
            jloc_pred.detach(),
            joff_pred.detach(),
            dis_pred.detach(),

            conf_thresh_init=conf_th,
            junc_thresh=junc_thresh,
            j2l_radius=self.j2l_radius_px,
            max_proposals=10000,

            line_conn_preds=line_conn_preds,    # ⭐ 这里真正用上 fc2
            seed_thresh=conf_th * 0.5,
            final_line_thresh=conf_th,
        )


        # ============================================================
        # 🧪 decode 结果统计 + 防止 lines_pred 为空
        # ============================================================
        B = jloc_pred.shape[0]
        device = jloc_pred.device

        # 将 None 替换为空 tensor，便于后续统计
        safe_lines_batch = []
        empty_count = 0
        total_lines = 0
        max_lines = 0
        min_lines = 10**9

        for b in range(B):
            lb = lines_batch[b]
            if lb is None:
                lb = torch.empty((0, 4), device=device, dtype=jloc_pred.dtype)
            # 统一到 (N,4)
            if lb.ndim == 1:
                lb = lb.view(-1, 4)
            elif lb.ndim == 2 and lb.size(-1) != 4:
                lb = lb.reshape(-1, 4)

            n_b = lb.shape[0]
            total_lines += n_b
            max_lines = max(max_lines, n_b)
            min_lines = min(min_lines, n_b)
            if n_b == 0:
                empty_count += 1
            safe_lines_batch.append(lb)

        # ---------- 6) 若整个 batch 都没有线段，做一次“保底解码重试” ----------
        fallback_used = False       # ✅ 新增：记录这一步有没有触发 fallback
        fallback_conf = None        # ✅ 新增：占个位，方便下面 if 判断
        fallback_junc = None

        if total_lines == 0:
            print("----------debug_整个 batch 都没有线段，做一次“保底解码重试----------------------")
            # 用极低阈值再 decode 一次，尽量多提出候选
            fallback_conf = conf_base * 0.5   # 比 base 还低
            fallback_junc = junc_base * 0.5

            # 1) 先用「纯几何」 decode 得到候选线段
            geo_lines_batch, geo_scores_batch = decode_lines_from_hafm(
                hafm_dense.detach(),
                jloc_pred.detach(),
                joff_pred.detach(),
                dis_pred.detach(),

                conf_thresh_init=fallback_conf,
                junc_thresh=fallback_junc,
                j2l_radius=j2l_radius * 1.5,
                max_proposals=10000,

                line_conn_preds=None,
                seed_thresh=fallback_conf * 0.5,
                final_line_thresh=fallback_conf,
            )

            # 2) 用 fc2 对候选线段打分
            line_conn_preds = self.compute_line_conn_scores(
                loi_features_thin,
                loi_features_aux,
                geo_lines_batch,
            )

            # 3) 再 decode 一次
            lines_batch, line_scores_batch = decode_lines_from_hafm(
                hafm_dense.detach(),
                jloc_pred.detach(),
                joff_pred.detach(),
                dis_pred.detach(),

                conf_thresh_init=fallback_conf,
                junc_thresh=fallback_junc,
                j2l_radius=j2l_radius * 1.5,
                max_proposals=10000,

                line_conn_preds=line_conn_preds,
                seed_thresh=fallback_conf * 0.5,
                final_line_thresh=fallback_conf,
            )

            fallback_used = True     # ✅ 标记这一步我们确实走了 fallback

        safe_lines_batch = []
        total_lines = 0
        empty_count = 0
        max_lines = 0
        min_lines = 10**9

        for b in range(B):
            lb = lines_batch[b]     
            if lb is None:
                lb = torch.empty((0, 4), device=device, dtype=jloc_pred.dtype)
            if lb.ndim == 1:
                lb = lb.view(-1, 4)
            elif lb.ndim == 2 and lb.size(-1) != 4:
                lb = lb.reshape(-1, 4)

            n_b = lb.shape[0]
            total_lines += n_b
            max_lines = max(max_lines, n_b)
            min_lines = min(min_lines, n_b)
            if n_b == 0:
                empty_count += 1
            safe_lines_batch.append(lb)

        # decode debug：保底重试信息
        if self.train_step % 200 == 0 and fallback_used:
            print(f"[decode-fallback] step={self.train_step} "
                f"| fallback_conf={fallback_conf:.4f}, fallback_junc={fallback_junc:.4f}, "
                f"total_lines={total_lines}, empty_imgs={empty_count}/{B}")


        # ---------- 7) 仍然有单张图像完全没有线段 -> 注入虚拟线段，避免 refinement 完全跳过 ----------
        # 说明：
        #  - 这些虚拟线段会在 refinement_train 中与 GT 做距离匹配，
        #    通常会被标成负样本，有助于训练“背景线段”的分类能力。
        Hf, Wf = jloc_pred.shape[2], jloc_pred.shape[3]
        dummy_injected = 0
        for b in range(B):
            if safe_lines_batch[b].shape[0] == 0:
                print("----------debug_仍然有单张图像完全没有线段 -> 注入虚拟线段，避免 refinement 完全跳过----------------------")
                # 在特征图中心插入一条对角线作为“保底线段”
                x1 = Wf * 0.25
                y1 = Hf * 0.25
                x2 = Wf * 0.75
                y2 = Hf * 0.75
                dummy_line = torch.tensor([[x1, y1, x2, y2]],
                                          device=device, dtype=jloc_pred.dtype)
                safe_lines_batch[b] = dummy_line
                dummy_injected += 1
                total_lines += 1
                max_lines = max(max_lines, 1)
                min_lines = min(min_lines, 1)

        # 最终用于 refinement 的 lines_batch
        lines_batch = safe_lines_batch

        # ---------- 8) decode + refine debug （每 500 step 打印一次） ----------
        if self.train_step % 500 == 0:
            mean_lines = total_lines / float(B) if B > 0 else 0.0
            if min_lines == 10**9:
                min_lines = 0
            print(
                f"[train-decode] step={self.train_step} | "
                f"conf_th={conf_th:.4f}, junc_th={junc_thresh:.4f}, "
                f"j2l_radius={j2l_radius:.1f} | "
                f"lines_per_img: mean={mean_lines:.1f}, min={min_lines}, max={max_lines}, "
                f"dummy_injected={dummy_injected}"
            )

        # ------------------ Refinement 训练 ------------------
        # 说明：
        #  - 经过上面的防空机制，至少每张图都会有 >=1 条线段进入 refinement_train
        #  - 即便是“虚拟线段”，也会被打上负标签，有助于训练分类器
        loss_dict_, extra_info_ = self.refinement_train(
            lines_batch, jloc_pred, joff_pred,
            loi_features, loi_features_thin, loi_features_aux, metas
        )

        loss_dict['loss_pos'] += loss_dict_['loss_pos']
        loss_dict['loss_neg'] += loss_dict_['loss_neg']
        loss_dict['loss_lineness'] += loss_dict_['loss_lineness']

        # ------------------ HAFM 主监督 ------------------
        mask = targets['mask']
        # ---- Safe mask fallback ----
        eps = 1e-6
        if torch.mean(mask) == 0:
            # 整张图没有正样本，直接兜底为全1，避免 /0
            mask = torch.ones_like(mask)
        den = torch.clamp(mask.mean(), min=eps)

        lines_tgt = self.hafm_decoding(
            targets['md'], targets['dis'], None,
            flatten=False, scale=self.hafm_encoder.dis_th
        )

        mask2 = []
        for i in range(batch_size):
            lines_gt = metas[i]['lines']
            temp = lines_tgt[i].reshape(-1, 4)
            temp_mask = torch.cdist(temp, lines_gt).min(dim=1)[0] < 1.0
            temp_mask = temp_mask.reshape(lines_tgt[i].shape[:-1])
            mask2.append(temp_mask)
        mask2 = torch.stack(mask2)

        lines_tgt = lines_tgt.repeat((1, 2 * self.use_residual + 1, 1, 1, 1))
        lines_len = torch.sum((lines_tgt[..., :2] - lines_tgt[..., 2:]) ** 2, dim=-1)

        # ------------------ 新增：HAFM 偏移场 L1 监督 ------------------
        if 'hafm' in targets:
            # targets['hafm']: [B,4,H,W]
            hafm_tgt = targets['hafm']
            # 与 hafm_pred_map 对齐
            if hafm_tgt.dim() == 3:
                hafm_tgt = hafm_tgt.unsqueeze(0)
            assert hafm_tgt.shape == hafm_pred_map.shape, \
                f"[hafm] tgt shape {hafm_tgt.shape} != pred shape {hafm_pred_map.shape}"

            # mask: [B,1,H,W] → 扩展到 4 通道
            mask_hafm = mask.expand_as(hafm_pred_map)  # [B,4,H,W]

            loss_map_hafm = F.l1_loss(
                hafm_pred_map, hafm_tgt, reduction='none'
            )
            loss_dict['loss_hafm'] += torch.mean(loss_map_hafm * mask_hafm) / den
        # ------------------ 多栈监督损失 ------------------
        if targets is not None:
            for nstack, output in enumerate(outputs):
                # ===== 先把输出分支提出来 =====
                md_out  = output[:, :3].sigmoid()   # [B,3,Hf,Wf]
                dis_out = output[:, 3:4].sigmoid()  # [B,1,Hf,Wf]
                res_out = output[:, 4:5].sigmoid()  # [B,1,Hf,Wf]

                H_tgt, W_tgt = targets['md'].shape[-2], targets['md'].shape[-1]

                # ---------- 统一上采样到 GT 分辨率 (128x128) ----------
                if md_out.shape[-2] != H_tgt or md_out.shape[-1] != W_tgt:
                    md_out = F.interpolate(
                        md_out, size=(H_tgt, W_tgt),
                        mode="bilinear", align_corners=False
                    )
                if dis_out.shape[-2] != targets['dis'].shape[-2] or dis_out.shape[-1] != targets['dis'].shape[-1]:
                    dis_out = F.interpolate(
                        dis_out, size=targets['dis'].shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                if res_out.shape[-2] != targets['res'].shape[-2] or res_out.shape[-1] != targets['res'].shape[-1]:
                    res_out = F.interpolate(
                        res_out, size=targets['res'].shape[-2:],
                        mode="bilinear", align_corners=False
                    )

                # ---------- 1. md 分支 ----------
                loss_map_md = torch.mean(
                    F.l1_loss(md_out, targets['md'], reduction='none'),
                    dim=1, keepdim=True
                )
                loss_dict['loss_md'] += torch.mean(loss_map_md * mask) / den

                # ---------- 2. dis 分支 ----------
                loss_map_dis = F.l1_loss(
                    dis_out, targets['dis'], reduction='none'
                )
                loss_dict['loss_dis'] += torch.mean(loss_map_dis * mask) / den

                # ---------- 3. res 分支 ----------
                loss_map_res = F.l1_loss(
                    res_out, targets['res'], reduction='none'
                )
                loss_dict['loss_res'] += torch.mean(loss_map_res * mask) / den

                # ---------- 4. junction heatmap ----------
                jloc_pred = output[:, 5:7]   # [B,2,Hf,Wf]
                H_tgt, W_tgt = targets['jloc'].shape[-2], targets['jloc'].shape[-1]

                # 如果预测尺寸（通常 64×64）与 target（默认 128×128）不一致，进行插值
                if jloc_pred.shape[-2] != H_tgt or jloc_pred.shape[-1] != W_tgt:
                    jloc_pred = F.interpolate(
                        jloc_pred, size=(H_tgt, W_tgt),
                        mode="bilinear", align_corners=False
                    )

                loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(
                    jloc_pred, targets['jloc']
                )

                joff_pred = output[:, 7:9]
                if joff_pred.shape[-2] != H_tgt or joff_pred.shape[-1] != W_tgt:
                    joff_pred = F.interpolate(
                        joff_pred, size=(H_tgt, W_tgt),
                        mode='bilinear', align_corners=False
                    )

                loss_dict['loss_joff'] += sigmoid_l1_loss(
                    joff_pred, targets['res'], -0.5, targets['jloc']
                )

                # ---------- 6. 辅助线段监督 (aux) ----------
                lines_learned = self.hafm_decoding(
                    output[:, :3].sigmoid(),
                    output[:, 3:4].sigmoid(),
                    output[:, 4:5].sigmoid() if self.use_residual else None,
                    flatten=False, scale=self.hafm_encoder.dis_th
                )
                # ====== AUX: 上采样到与 target 一致的 128×128 ======
                H_tgt, W_tgt = lines_tgt.shape[-3], lines_tgt.shape[-2]  # 例如 128,128

                if lines_learned.shape[-3] != H_tgt or lines_learned.shape[-2] != W_tgt:
                    # lines_learned: [B, C_dir, Hs, Ws, 4]
                    B, C_dir, Hs, Ws, P = lines_learned.shape

                    # 先把 (C_dir,4) 合并到通道维度，变成标准 4D 张量再插值
                    lines_learned_4d = lines_learned.view(B, C_dir * P, Hs, Ws)  # [B, C_dir*4, Hs, Ws]

                    lines_learned_4d = F.interpolate(
                        lines_learned_4d,
                        size=(H_tgt, W_tgt),
                        mode="bilinear",
                        align_corners=False
                    )  # [B, C_dir*4, H_tgt, W_tgt]

                    # 再还原回 [B, C_dir, H_tgt, W_tgt, 4]
                    lines_learned = lines_learned_4d.view(B, C_dir, H_tgt, W_tgt, P)

                wt = 1 / lines_len.clamp_min(1.0) * mask2
                # ✅ 这里的 loss_map 与上面的不同，是 aux 分支的局部变量
                loss_map_aux = F.l1_loss(
                    lines_learned, lines_tgt, reduction='none'
                ).mean(dim=-1)
                loss_dict['loss_aux'] += torch.mean(loss_map_aux * wt) / torch.mean(mask)

        for key in extra_info_.keys():
            extra_info[key] = extra_info_[key] / batch_size
        
        return loss_dict, extra_info


    def hafm_decoding_mask(self, md_maps, dis_maps, residual_maps, scores, scale=5.0):
        device = md_maps.device

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0, x0 =torch.meshgrid(_y, _x,indexing='ij')
        y0 = y0.reshape(1,1,-1)
        x0 = x0.reshape(1,1,-1)
        
        sign_pad = torch.arange(-self.use_residual,self.use_residual+1,device=device,dtype=torch.float32).reshape(1,-1,1)

        if residual_maps is not None:
            residual = residual_maps.reshape(batch_size,1,-1)*sign_pad
            distance_fields = dis_maps.reshape(batch_size,1,-1) + residual
            scores = scores.reshape(batch_size,1,-1).repeat((1,2*self.use_residual+1,1))
        else:
            distance_fields = dis_maps.reshape(batch_size,1,-1)
            scores = scores.reshape(batch_size,1,-1)
        md_maps = md_maps.reshape(batch_size,3,-1)
        
        distance_fields = distance_fields.clamp(min=0,max=1.0)
        md_un = (md_maps[:,:1] - 0.5)*np.pi*2
        st_un = md_maps[:,1:2]*np.pi/2.0
        ed_un = -md_maps[:,2:3]*np.pi/2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md*y_st)*distance_fields*scale
        y_st_rotated = (ss_md + cs_md*y_st)*distance_fields*scale

        x_ed_rotated = (cs_md - ss_md*y_ed)*distance_fields*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)*distance_fields*scale

        x_st_final = (x_st_rotated + x0).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0,max=height-1)

        
        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final),dim=-1)

        lines = lines.reshape(batch_size,-1,4)
        scores = scores.reshape(batch_size,-1)
        
        sc_, arg_ = scores[0].sort(descending=True)
        lines_out = lines[0][arg_[sc_>0]]
        
        return lines_out, sc_[sc_>0]

    def hafm_decoding(self, md_maps, dis_maps, residual_maps, scale=5.0, flatten=True):
        device = md_maps.device
        batch_size, _, height, width = md_maps.shape
    
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()
        y0, x0 = torch.meshgrid(_y, _x, indexing='ij')
        y0, x0 = y0[None, None], x0[None, None]
    
        sign_pad = torch.arange(-self.use_residual, self.use_residual + 1,
                                device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    
        # ===========================================================
        # ✅ residual/dis 对齐：residual→(B,3,H,W)，dis→(B,3,H,W)；否则 dis→(B,1,H,W)
        # ===========================================================
        if residual_maps is not None:
            # residual_maps 常见为 (B,2,H,W)（x/y 偏移），也可能是 (B,1,H,W)
            if residual_maps.dim() != 4:
                raise ValueError(f"[error] unexpected residual_maps shape: {residual_maps.shape}")
            if residual_maps.size(1) == 2:
                # (B,2,H,W) * (1,3,1,1) → (B,2,3,H,W) → 平均 x/y → (B,3,H,W)
                residual = residual_maps.unsqueeze(2) * sign_pad.unsqueeze(1)
                residual = residual.mean(1)
            elif residual_maps.size(1) == 1:
                # (B,1,H,W) * (1,3,1,1) → (B,3,H,W)
                residual = residual_maps * sign_pad
            else:
                # 有 3 个或更多通道时，裁成 3 个用于三方向
                residual = residual_maps[:, :3]
    
            # dis_maps：与 residual 通道对齐成 (B,3,H,W)
            if dis_maps.dim() != 4:
                raise ValueError(f"[error] unexpected dis_maps shape: {dis_maps.shape}")
            if dis_maps.size(1) == 1:
                dis_maps_rep = dis_maps.repeat(1, residual.size(1), 1, 1)
            else:
                dis_maps_rep = dis_maps[:, :residual.size(1)]
            distance_fields = (dis_maps_rep + residual).clamp(min=0.0, max=1.0)
        else:
            # 无 residual：dis 保持 (B,1,H,W)
            if dis_maps.dim() != 4:
                raise ValueError(f"[error] unexpected dis_maps shape: {dis_maps.shape}")
            distance_fields = dis_maps.clamp(min=0.0, max=1.0)
    
        # ===========================================================
        # ✅ md 通道兜底：保证 md/st/ed 三个通道都存在（缺的补 0）
        # ===========================================================
        C_md = md_maps.size(1)
        md0 = md_maps[:, :1]                                     # 一定存在
        st_ch = md_maps[:, 1:2] if C_md >= 2 else torch.zeros_like(md0)
        ed_ch = md_maps[:, 2:3] if C_md >= 3 else torch.zeros_like(md0)
    
        md_un = (md0 - 0.5) * (np.pi * 2.0)                      # [-pi, pi]
        st_un = st_ch * (np.pi / 2.0)
        ed_un = -ed_ch * (np.pi / 2.0)
    
        cs_md, ss_md = md_un.cos(), md_un.sin()
        y_st, y_ed = torch.tan(st_un), torch.tan(ed_un)
    
        # --- 广播：distance_fields 可能为 (B,1,H,W) 或 (B,3,H,W)；与 (B,1,H,W) 自动对齐 ---
        x_st_rotated = (cs_md - ss_md * y_st) * distance_fields * scale
        y_st_rotated = (ss_md + cs_md * y_st) * distance_fields * scale
        x_ed_rotated = (cs_md - ss_md * y_ed) * distance_fields * scale
        y_ed_rotated = (ss_md + cs_md * y_ed) * distance_fields * scale
    
        x_st_final = (x_st_rotated + x0).clamp(min=0, max=width - 1)
        y_st_final = (y_st_rotated + y0).clamp(min=0, max=height - 1)
        x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width - 1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height - 1)
    
        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final), dim=-1)
        if flatten:
            lines = lines.reshape(batch_size, -1, 4)
    
        # if batch_size > 0:
        #     print(f"[debug] distance_fields={tuple(distance_fields.shape)}, md_maps={tuple(md_maps.shape)}")
    
        return lines


def get_hawp_model(pretrained = False):
    from parsing.config import cfg
    import os
    model = WireframeDetector(cfg)
    if pretrained:
        url = PRETRAINED.get('url')
        hubdir = torch.hub.get_dir()
        filename = os.path.basename(url)
        dst = os.path.join(hubdir,filename)
        state_dict = torch.hub.load_state_dict_from_url(url,dst)
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
    return model

        
