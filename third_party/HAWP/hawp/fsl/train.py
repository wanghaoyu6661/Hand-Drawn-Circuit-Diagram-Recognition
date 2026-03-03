import torch
import random
import numpy as np
import os
# ===== NumPy 兼容性补丁：修复 metric_evaluation 里用到的 np.float =====
if not hasattr(np, "float"):
    np.float = float
import os.path as osp
import time
import datetime
import argparse
import logging
import json
import math
from tqdm import tqdm

import hawp
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.metric_logger import MetricLogger
from hawp.base.utils.miscellaneous import save_config
from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.metric_evaluation import TPFP, AP

from hawp.fsl.dataset import build_train_dataset, build_test_dataset
from hawp.fsl.config import cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.model.build import build_model
from hawp.fsl.solver import make_lr_scheduler, make_optimizer

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_output_dir(root, basename):
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return os.path.join(root, basename, timestamp)

def compute_sap(result_list, annotations_dict, threshold):
    tp_list, fp_list, scores_list = [], [], []
    n_gt = 0

    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]

        lines_pred = np.array(res['lines_pred'], dtype=np.float32)
        scores = np.array(res['lines_score'], dtype=np.float32)

        if lines_pred.size == 0:
            # 没有预测线段，直接跳过；后面再统一处理
            continue

        # 按分数排序（降序）
        sort_idx = np.argsort(-scores)
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]

        # 统一缩放到 128x128
        lines_pred[:, 0] *= 128.0 / float(res['width'])
        lines_pred[:, 1] *= 128.0 / float(res['height'])
        lines_pred[:, 2] *= 128.0 / float(res['width'])
        lines_pred[:, 3] *= 128.0 / float(res['height'])

        lines_gt = np.array(gt['lines'], dtype=np.float32)
        lines_gt[:, 0] *= 128.0 / float(gt['width'])
        lines_gt[:, 1] *= 128.0 / float(gt['height'])
        lines_gt[:, 2] *= 128.0 / float(gt['width'])
        lines_gt[:, 3] *= 128.0 / float(gt['height'])

        # 累加全局 GT 数量（✅ 修复）
        n_gt += lines_gt.shape[0]

        tp, fp = TPFP(lines_pred, lines_gt, threshold)
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

    # ==== 极端情况保护：整个 val 集都没有预测 ====
    if len(tp_list) == 0 or len(scores_list) == 0 or n_gt == 0:
        return 0.0, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)

    # 全局按分数排序
    idx = np.argsort(scores_list)[::-1]
    tp_cum = np.cumsum(tp_list[idx])
    fp_cum = np.cumsum(fp_list[idx])

    # ✅ 正确的 recall / precision
    rcs = tp_cum / float(n_gt)
    pcs = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # ✅ 把 (recall, precision) 传给 AP
    sAP = AP(rcs, pcs) * 100.0

    return sAP, pcs, rcs



class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    
    def __call__(self, loss_dict, model=None):
        total = 0.0
        # 优先使用动态更新后的权重（若存在）
        active_weights = getattr(model, "loss_weights", self.loss_weights)
        for k, w in active_weights.items():
            if k not in loss_dict:
                continue
            total += w * loss_dict[k]
        return total

def train(cfg, model, train_dataset, val_datasets, optimizer, scheduler, loss_reducer, checkpointer, arguments):
    import math
    logger = logging.getLogger("hawp.trainer")
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments["epoch"]
    num_epochs = arguments["max_epoch"] - start_epoch
    epoch_size = len(train_dataset)

    total_iterations = num_epochs * epoch_size
    step = 0

    # ============================
    # 新增：全局训练步数
    # ============================
    if not hasattr(model, "train_step"):
        model.train_step = 0

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        model.train()
        # ✅ 将当前 epoch 传入模型（forward_train 内可读取）
        model.current_epoch = epoch
        # ============================================================
        # ⭐ 三阶段动态调度（前期冻结 → 中期余弦下降 → 后期固定）
        # ============================================================
        ph_cfg = cfg.MODEL.PARSING_HEAD
        enc_cfg = cfg.ENCODER
        loss_cfg = cfg.MODEL.LOSS_WEIGHTS
        dset_cfg = cfg.DATASETS

        j2l_start  = getattr(ph_cfg, "J2L_THRESHOLD_START", 100.0)
        j2l_end    = getattr(ph_cfg, "J2L_THRESHOLD_END",   80.0)
        pos_start  = getattr(ph_cfg, "POS_MATCH_THRESHOLD_START", 100.0)
        pos_end    = getattr(ph_cfg, "POS_MATCH_THRESHOLD_END",   80.0)
        dist_start = getattr(dset_cfg, "DISTANCE_TH_START", 0.04)
        dist_end   = getattr(dset_cfg, "DISTANCE_TH_END",   0.025)

        # 总 epoch 进度（0 ~ 1）
        progress = epoch / float(cfg.SOLVER.MAX_EPOCH)
        progress = min(progress, 1.0)

        # 三阶段边界
        hold_end  = 0.30   # 0%~30%：冻结
        decay_end = 0.90   # 30%~90%：下降
        # 90%~100%：固定为 end

        def three_stage_interp(start, end, progress):
            """
            三阶段插值：
            - [0, 0.30): 冻结
            - [0.30, 0.90): 余弦平滑下降
            - [0.90, 1.00]: 固定 end
            """
            if progress < hold_end:
                return start

            elif progress < decay_end:
                # 缩放到 0~1 范围
                p = (progress - hold_end) / (decay_end - hold_end)
                return start + (end - start) * 0.5 * (1 - math.cos(math.pi * p))

            else:
                return end

        # ============================================================
        # ⭐ 1) J2L_THRESHOLD 动态调度
        # ============================================================
        j2l_current = three_stage_interp(j2l_start, j2l_end, progress)
        model.j2l_radius_px = j2l_current

        logger.info(
            f"[sched] epoch={epoch:03d} | J2L_THRESHOLD={j2l_current:.1f}px "
            f"(from {j2l_start}→{j2l_end})"
        )

        # ============================================================
        # ⭐ 2) POS_MATCH_THRESHOLD 动态调度
        # ============================================================
        pos_current = three_stage_interp(pos_start, pos_end, progress)
        model.pos_match_radius_px = pos_current

        logger.info(f"[sched] epoch={epoch:03d} | POS_MATCH_TH={pos_current:.1f}px")

        # ============================================================
        # ⭐ 3) DISTANCE_TH 动态调度
        # ============================================================
        dist_current = three_stage_interp(dist_start, dist_end, progress)
        model.distance_th = dist_current

        logger.info(f"[sched] epoch={epoch:03d} | DISTANCE_TH={dist_current:.4f}")

        # ============================================================
        # ⭐ 4) 其它动态参数同步三阶段调度（必须同步，否则训练不稳定）
        # ============================================================

        # MAX_DISTANCE
        if hasattr(model, "max_distance"):
            model.max_distance = three_stage_interp(
                getattr(ph_cfg, "MAX_DISTANCE_START", 15.0),
                getattr(ph_cfg, "MAX_DISTANCE_END",   12.0),
                progress
            )

        # N_DYN_JUNC
        model.n_dyn_junc = int(three_stage_interp(
            getattr(ph_cfg, "N_DYN_JUNC_START", 300),
            getattr(ph_cfg, "N_DYN_JUNC_END",   200),
            progress
        ))

        # N_DYN_POSL
        model.n_dyn_posl = int(three_stage_interp(
            getattr(ph_cfg, "N_DYN_POSL_START", 60),
            getattr(ph_cfg, "N_DYN_POSL_END",   80),
            progress
        ))

        # N_DYN_NEGL
        model.n_dyn_negl = int(three_stage_interp(
            getattr(ph_cfg, "N_DYN_NEGL_START", 25),
            getattr(ph_cfg, "N_DYN_NEGL_END",   15),
            progress
        ))

        # 背景权重
        model.background_weight = three_stage_interp(
            getattr(enc_cfg, "BACKGROUND_WEIGHT_START", 0.1),
            getattr(enc_cfg, "BACKGROUND_WEIGHT_END",   0.05),
            progress
        )

        # loss 权重组
        model.loss_weights = {
            "loss_jloc": three_stage_interp(getattr(loss_cfg, "loss_jloc_START", 3.0), getattr(loss_cfg, "loss_jloc_END", 2.0), progress),
            "loss_joff": three_stage_interp(getattr(loss_cfg, "loss_joff_START", 0.25), getattr(loss_cfg, "loss_joff_END", 0.15), progress),
            "loss_md":   three_stage_interp(getattr(loss_cfg, "loss_md_START", 0.5), getattr(loss_cfg, "loss_md_END", 0.3), progress),
            "loss_res":  three_stage_interp(getattr(loss_cfg, "loss_res_START", 0.5), getattr(loss_cfg, "loss_res_END", 0.8), progress),
            "loss_pos":  three_stage_interp(getattr(loss_cfg, "loss_pos_START", 1.0), getattr(loss_cfg, "loss_pos_END", 0.5), progress),
            "loss_neg":  three_stage_interp(getattr(loss_cfg, "loss_neg_START", 1.0), getattr(loss_cfg, "loss_neg_END", 0.5), progress),
        }

        logger.info(
            f"[sched] epoch={epoch:03d} | J2L={model.j2l_radius_px:.1f} | "
            f"MAX_D={model.max_distance:.1f} | N_JUNC={model.n_dyn_junc} | "
            f"loss_jloc={model.loss_weights['loss_jloc']:.2f}"
        )

        loss_meters = MetricLogger(" ")
        aux_meters = MetricLogger(" ")
        sys_meters = MetricLogger(" ")

        for it, (images, annotations) in enumerate(train_dataset):

            model.train_step += 1

            data_time = time.time() - end
            images = images.to(device)
            annotations = to_device(annotations, device)

            optimizer.zero_grad()

            # ========== AMP 自动混合精度 ==========
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss_dict, extra_info = model(images, annotations)
                total_loss = loss_reducer(loss_dict, model)

            # ========== 记录损失（不影响反向传播） ==========
            with torch.no_grad():
                loss_reduced = total_loss.item()
                loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}

                loss_meters.update(loss=loss_reduced, **loss_dict_reduced)
                aux_meters.update(**extra_info)

            # ========== AMP Backward ==========
            scaler.scale(total_loss).backward()

            # ✅ AMP 下：先把梯度 unscale 回正常尺度，再裁剪
            scaler.unscale_(optimizer)
            # ========== 梯度裁剪 ==========
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # ========== AMP Step ==========
            scaler.step(optimizer)
            scaler.update()

            # ========== 学习率调度（你的原逻辑保留） ==========
            model.global_step = getattr(model, "global_step", 0) + 1

            base_lr = 1e-5
            max_lr = 2e-5
            warmup_iters = 1000
            total_iters = len(train_dataset) * cfg.SOLVER.MAX_EPOCH

            if model.global_step < warmup_iters:
                lr = base_lr + (max_lr - base_lr) * (model.global_step / warmup_iters)
            else:
                progress = (model.global_step - warmup_iters) / (total_iters - warmup_iters)
                lr = max_lr * 0.5 * (1 - math.cos(math.pi * min(1.0, progress)))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ========== 计时、日志 ==========
            batch_time = time.time() - end
            end = time.time()
            sys_meters.update(time=batch_time, data=data_time)

            total_iterations -= 1
            step += 1

            eta_seconds = sys_meters.time.global_avg * total_iterations
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if it % 20 == 0 or it + 1 == len(train_dataset):
                logger.info(
                    "eta: {eta} epoch: {epoch} iter: {iter} lr: {lr:.6f} max mem: {memory:.0f}\n"
                    "RUNTIME: {sys_meters}\n"
                    "LOSSES: {loss_meters}\n"
                    "AUXINFO: {aux_meters}\n"
                    "WorkingDIR: {wdir}\n".format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        lr=lr,
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        sys_meters=str(sys_meters),
                        loss_meters=str(loss_meters),
                        aux_meters=str(aux_meters),
                        wdir=cfg.OUTPUT_DIR,
                    )
                )


        save_last_n = int(getattr(cfg.MODEL, "SAVE_LAST_N", 20))
        last_models_dir = os.path.join(cfg.OUTPUT_DIR, "last_epochs")
        os.makedirs(last_models_dir, exist_ok=True)

        # 2) 保存当前 epoch 的模型
        last_ckpt_path = os.path.join(last_models_dir, f"last_epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, last_ckpt_path)
        logger.info(f"[last-save] Saved epoch-{epoch} model → {last_ckpt_path}")

        # 3) 删除多余的旧模型，保持目录中最多 N 个
        saved_files = sorted(os.listdir(last_models_dir))
        if len(saved_files) > save_last_n:
            remove_count = len(saved_files) - save_last_n
            for f in saved_files[:remove_count]:
                try:
                    os.remove(os.path.join(last_models_dir, f))
                    logger.info(f"[last-save] removed old checkpoint: {f}")
                except:
                    pass

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / num_epochs
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAWPv2 Training')

    parser.add_argument("config",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument('--logdir', required=True, type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')
    parser.add_argument("--seed",
                        default=42,
                        type=int)
    parser.add_argument('--tf32', default=False, action='store_true', help='toggle on the TF32 of pytorch')
    parser.add_argument('--dtm', default=True, choices=[True, False], help='toggle the deterministic option of CUDNN. This option will affect the replication of experiments')

    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.dtm

    assert args.config.endswith('yaml') or args.config.endswith('yml')
    config_basename = os.path.basename(args.config)
    if config_basename.endswith('yaml'):
        config_basename = config_basename[:-5]
    else:
        config_basename = config_basename[:-4]

    # ================================================================
    # 🔧 手动注入自定义动态参数（避免 yacs KeyError）
    # ================================================================
    custom_dataset_keys = [
        "DISTANCE_TH_START", "DISTANCE_TH_END"
    ]
    custom_encoder_keys = [
        "BACKGROUND_WEIGHT_START", "BACKGROUND_WEIGHT_END"
    ]
    custom_loss_keys = [
        "loss_jloc_START", "loss_jloc_END",
        "loss_joff_START", "loss_joff_END",
        "loss_md_START", "loss_md_END",
        "loss_res_START", "loss_res_END"
    ]
    custom_parsing_keys = [
        "J2L_THRESHOLD_START", "J2L_THRESHOLD_END"
    ]
    for key in custom_dataset_keys:
        if key not in cfg.DATASETS:
            cfg.DATASETS[key] = None
    for key in custom_encoder_keys:
        if key not in cfg.ENCODER:
            cfg.ENCODER[key] = None
    for key in custom_loss_keys:
        if key not in cfg.MODEL.LOSS_WEIGHTS:
            cfg.MODEL.LOSS_WEIGHTS[key] = None
    for key in custom_parsing_keys:
        if key not in cfg.MODEL.PARSING_HEAD:
            cfg.MODEL.PARSING_HEAD[key] = None

    cfg.merge_from_file(args.config)

    output_dir = get_output_dir(args.logdir, config_basename)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir)
    
    logger = setup_logger('hawp', output_dir, out_file='train.log')

    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config))

    with open(args.config,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    # ====== AMP 混合精度 ======
    scaler = torch.cuda.amp.GradScaler()

    loss_reducer = LossReducer(cfg)

    arguments = {}
    arguments["epoch"] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         optimizer,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        logger.info('loading the pretrained model from {}'.format(args.resume))
        
    train_dataset = build_train_dataset(cfg)
    val_datasets = build_test_dataset(cfg)
    logger.info('epoch size = {}'.format(len(train_dataset)))
    train(cfg, model, train_dataset , val_datasets, optimizer, scheduler, loss_reducer, checkpointer, arguments)    

    import pdb; pdb.set_trace()
