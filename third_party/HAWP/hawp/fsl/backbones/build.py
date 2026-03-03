from .registry import MODELS
from .stacked_hg import HourglassNet, Bottleneck2D
from .multi_task_head import MultitaskHead
from .resnets import ResNets


# =========================================================
# 1) Hourglass backbone (默认)
# =========================================================
@MODELS.register("Hourglass")
def build_hg(cfg, **kwargs):

    inplanes = cfg.MODEL.HGNETS.INPLANES
    num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS // 2
    depth = cfg.MODEL.HGNETS.DEPTH
    num_stacks = cfg.MODEL.HGNETS.NUM_STACKS
    num_blocks = cfg.MODEL.HGNETS.NUM_BLOCKS
    head_size = cfg.MODEL.HEAD_SIZE
    out_feature_channels = cfg.MODEL.OUT_FEATURE_CHANNELS

    input_channels = 1 if kwargs.get("gray_scale", False) else 3
    num_class = sum(sum(head_size, []))

    model = HourglassNet(
        input_channels=input_channels,
        block=Bottleneck2D,
        inplanes=inplanes,
        num_feats=num_feats,
        depth=depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_classes=num_class
    )

    model.out_feature_channels = out_feature_channels
    return model


# =========================================================
# 2) ResNet50 backbone (ImageNet 预训练)
# =========================================================
@MODELS.register("ResNets")
def build_resnet(cfg, **kwargs):

    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))

    basenet = cfg.MODEL.RESNETS.BASENET      # resnet50
    use_pretrain = cfg.MODEL.RESNETS.PRETRAIN  # true / false

    model = ResNets(
        basenet,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_class=num_class,
        pretrain=use_pretrain,
    )

    # PixelShuffle 下采样后通道维度固定为 128
    model.out_feature_channels = 128
    print(f"[Backbone] ResNet built. Pretrain={use_pretrain}, out_channels=128")

    return model



# =========================================================
# 3) 总调度接口：根据 cfg.MODEL.NAME 创建 backbone
# =========================================================
def build_backbone(cfg, **kwargs):

    name = cfg.MODEL.NAME
    assert name in MODELS, f"cfg.MODEL.NAME='{name}' 未注册. 支持: {list(MODELS.keys())}"

    print(f"[Backbone] Using backbone: {name}")
    return MODELS[name](cfg, **kwargs)
