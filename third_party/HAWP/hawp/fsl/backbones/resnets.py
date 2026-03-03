import torch
import torch.nn as nn
import torchvision


class ResNets(nn.Module):
    RESNET_TEMPLATES = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
    }

    def __init__(self, basenet, head, num_class, pretrain=True):
        """
        Ultra 版 backbone：
        - 保留 ResNet 结构和 PixelShuffle(4)
        - 通过减小前端 stride，把整体下采样比例从 32 降到 8
        - PixelShuffle ×4 后，最终 stride ≈ 2（更高的空间分辨率）
        """
        super(ResNets, self).__init__()
        assert basenet in ResNets.RESNET_TEMPLATES

        basenet_fn = ResNets.RESNET_TEMPLATES.get(basenet)

        # 兼容旧版本 torchvision：pretrain 仍作为 "pretrained" 传入
        # 如果你的环境很新，可以改成 weights=xxx 的写法，这里先保持你原来的风格
        model = basenet_fn(pretrain)

        # ===========================
        # ★ Ultra 版关键修改点 ★
        # ===========================
        # 原来是：Conv2d(3, 64, 7, stride=2, padding=3)
        #   会在一开始就 /2，后面再配合 MaxPool /2，过早丢失空间分辨率。
        #
        # 现在改成 stride=1：不在这里做下采样。
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 原来是：MaxPool2d(3, stride=2, padding=1)
        #   再次 /2，导致到 layer1 之前总共 /4。
        #
        # Ultra 版改为 stride=1，相当于“去掉”这个下采样，只做 3x3 的平滑汇聚。
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 后面几层保持和 torchvision 的 ResNet 一致（仍然在 layer2/3/4 里做 stride=2）
        self.layer1 = model.layer1  # 不下采样
        self.layer2 = model.layer2  # /2
        self.layer3 = model.layer3  # /2
        self.layer4 = model.layer4  # /2

        # ===========================
        # 空间上采样：PixelShuffle(4)
        # ===========================
        # 以 ResNet50 为例，layer4 输出通道 2048：
        #   2048 / (4*4) = 128 通道
        # 于是 PixelShuffle(4) 后变成 (128, H*4, W*4)
        self.pixel_shuffle = nn.PixelShuffle(4)

        # HAWP 的 hafm head，输入通道数 128（对应 PixelShuffle 后的通道）
        self.hafm_predictor = head(128, num_class)

    def forward(self, images):
        """
        输入: images (B, 3, H, W)，一般 H=W=512
        输出:
          - [hafm_pred]:     list，长度 1，元素为 (B, num_class, H_out, W_out)
          - features:        PixelShuffle 之前的特征（给 HAWP 其它头用）
        最终空间 stride ≈ 2：
          H_out ≈ H / 2, W_out ≈ W / 2
        """
        # stem
        x = self.conv1(images)      # stride=1, 形状仍为 H, W
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)         # stride=1, 形状仍为 H, W

        # ResNet 主体
        x = self.layer1(x)          # 依然保持 H, W
        x = self.layer2(x)          # /2
        x = self.layer3(x)          # /4
        x = self.layer4(x)          # /8

        x = self.pixel_shuffle(x)
        features = x                  # 让后续 fc1 用当前 128通道特征
        hafm_pred = self.hafm_predictor(x)
        return [hafm_pred], features



if __name__ == "__main__":
    # 简单自测（你可以在容器里跑一下）
    from types import SimpleNamespace

    class DummyHead(nn.Module):
        def __init__(self, in_channels, num_class):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, num_class, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    model = ResNets('resnet50', head=DummyHead, num_class=8, pretrain=False)
    inp = torch.zeros((1, 3, 512, 512))
    outs, feats = model(inp)
    print("features shape (before PixelShuffle):", feats.shape)
    print("hafm_pred shape:", outs[0].shape)
