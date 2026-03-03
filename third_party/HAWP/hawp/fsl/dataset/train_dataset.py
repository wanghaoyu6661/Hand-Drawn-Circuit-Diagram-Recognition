# hawp/fsl/dataset/train_dataset_fixed.py
import json, copy, os.path as osp
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torch.utils.data.dataloader import default_collate
import os


class TrainDataset(Dataset):
    """
    修正版：
    - 将 0~1 归一化坐标乘回 (width, height) -> 像素坐标
    - 对 edges_negative 为空的样本做安全处理（空 (0,2) 数组）
    - 保留原有 4/5 种增强（hflip / vflip / hvflip / 90°旋转）
    - 保证 image 为 HxWx3 float32
    """
    def __init__(self, root, ann_file, transform=None, augmentation=4):
        self.root = root
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        # 与原实现一致，允许 idx_ // len 的增广循环
        return len(self.annotations) * max(1, self.augmentation)

    def __getitem__(self, idx_):
        idx = idx_ % len(self.annotations)
        reminder = idx_ // len(self.annotations)  # 0..augmentation-1
        ann = copy.deepcopy(self.annotations[idx])

        # ---- 读图，三通道 ----
        img_path = osp.join(self.root, ann['filename'])
        image = io.imread(img_path).astype(np.float32)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        else:
            image = image[:, :, :3]

        # ---- numpy 化 ----
        for key, dtype in (('junctions', np.float32),
                           ('edges_positive', np.int32),
                           ('edges_negative', np.int32)):
            ann[key] = np.array(ann.get(key, []), dtype=dtype)

        width  = int(ann['width'])
        height = int(ann['height'])

        # ---- 归一化坐标 -> 像素坐标（关键修复）----
        if ann['junctions'].size > 0 and ann['junctions'].max() <= 1.0:
            ann['junctions'][:, 0] *= width
            ann['junctions'][:, 1] *= height

        # ---- 空负样本安全处理（关键修复）----
        if ann['edges_negative'].size == 0:
            ann['edges_negative'] = np.zeros((0, 2), dtype=np.int32)

        # ---- 数据增强（与原版一致）----
        # 0: none; 1: hflip; 2: vflip; 3: hvflip; 4: rot90
        if reminder == 1:
            image = image[:, ::-1, :]
            if ann['junctions'].size > 0:
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
        elif reminder == 2:
            image = image[::-1, :, :]
            if ann['junctions'].size > 0:
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
        elif reminder == 3:
            image = image[::-1, ::-1, :]
            if ann['junctions'].size > 0:
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
        elif reminder == 4:
            # 逆时针 90°
            image_rot = np.rot90(image)
            if ann['junctions'].size > 0:
                # 以图像中心为原点旋转
                cx, cy = width / 2.0, height / 2.0
                pts = ann['junctions'] - np.array([[cx, cy]], dtype=np.float32)
                theta = 0.5 * np.pi
                rot = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]], dtype=np.float32)
                pts_r = pts @ rot.T
                h2, w2 = image_rot.shape[0], image_rot.shape[1]
                ann['junctions'] = pts_r + np.array([[w2 / 2.0, h2 / 2.0]], dtype=np.float32)
            image = image_rot
            ann['width'], ann['height'] = image.shape[1], image.shape[0]
            width, height = ann['width'], ann['height']

                # —— 在 __getitem__ 返回前加（可用环境变量控制开关）——
        
        if os.environ.get("HAWP_DEBUG", "0") == "1":
            j = ann['junctions']
            ep = ann.get('edges_positive', [])
            en = ann.get('edges_negative', [])
            j_np = np.asarray(j)
            
        # --------------------------------------------------------
        # 🔧 自动补全 ann["lines"]（训练集 & 验证集通用）
        # --------------------------------------------------------
        if "lines" not in ann or len(ann["lines"]) == 0:
            lines = []
            juncs = ann["junctions"]
            edges = ann.get("edges_positive", [])
        
            # edges 是 junc index 对，例如 [[1,5],[2,3],...]
            for e in edges:
                j1 = juncs[e[0]]
                j2 = juncs[e[1]]
                lines.append([
                    float(j1[0]), float(j1[1]),
                    float(j2[0]), float(j2[1])
                ])
        
            ann["lines"] = lines

        # ---- 返回 transform(image, ann) 或原始 ----
        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])
