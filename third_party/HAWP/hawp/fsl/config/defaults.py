from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER
from .detr import DETR

cfg = CN()

# ================================================================
# ⚙️ ENCODER
# ================================================================
cfg.ENCODER = CN()
cfg.ENCODER.DIS_TH = 5
cfg.ENCODER.ANG_TH = 0.1
cfg.ENCODER.NUM_STATIC_POS_LINES = 300
cfg.ENCODER.NUM_STATIC_NEG_LINES = 40
cfg.ENCODER.BACKGROUND_WEIGHT = 0.0
cfg.ENCODER.BACKGROUND_WEIGHT_START = 0.10
cfg.ENCODER.BACKGROUND_WEIGHT_END = 0.05


# ================================================================
# ⚙️ MODEL
# ================================================================
cfg.MODEL = MODELS
cfg.MODELING_PATH = "hawp"
cfg.MODEL.MIN_VALID_PAIRS = 2

# ✅ 确保 PARSING_HEAD 节点存在
if not hasattr(cfg.MODEL, "PARSING_HEAD"):
    cfg.MODEL.PARSING_HEAD = CN()

# ✅ 动态参数配置（用于 train.py 动态调度）
cfg.MODEL.PARSING_HEAD.POS_MATCH_THRESHOLD_START = 100.0
cfg.MODEL.PARSING_HEAD.POS_MATCH_THRESHOLD_END = 30.0

cfg.MODEL.PARSING_HEAD.N_DYN_JUNC_START = 300
cfg.MODEL.PARSING_HEAD.N_DYN_JUNC_END = 200
cfg.MODEL.PARSING_HEAD.N_DYN_POSL_START = 60
cfg.MODEL.PARSING_HEAD.N_DYN_POSL_END = 80
cfg.MODEL.PARSING_HEAD.N_DYN_NEGL_START = 25
cfg.MODEL.PARSING_HEAD.N_DYN_NEGL_END = 15
cfg.MODEL.PARSING_HEAD.N_DYN_POSL_ENDPOS_MATCH_THRESHOLD_START = 0.0

# 其他动态参数（建议一并补齐）
cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD_START = 100.0
cfg.MODEL.PARSING_HEAD.J2L_THRESHOLD_END = 25.0
cfg.MODEL.PARSING_HEAD.MAX_DISTANCE_START = 15.0
cfg.MODEL.PARSING_HEAD.MAX_DISTANCE_END = 5.0

# ================================================================
# ⚙️ 其他模块
# ================================================================
cfg.DATASETS = DATASETS
cfg.SOLVER = SOLVER

cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.DATASETS.DISTANCE_TH_START = 0.04
cfg.DATASETS.DISTANCE_TH_END = 0.025

cfg.OUTPUT_DIR = "outputs/dev"
