import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import json
import copy
from PIL import Image
from skimage import io
import os
import os.path as osp
import numpy as np
import cv2
class TestDatasetWithAnnotations(Dataset):
    '''
    Format of the annotation file
    annotations[i] has the following dict items:
    - filename  # of the input image, str 
    - height    # of the input image, int
    - width     # of the input image, int
    - lines     # of the input image, list of list, N*4
    - junc      # of the input image, list of list, M*2
    '''

    def __init__(self, root, ann_file, transform = None):
        self.root = root
        with open(ann_file, 'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = io.imread(osp.join(self.root, ann['filename']))
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=-1)
    
        image = image.astype(np.float32)[:, :, :3]
        h, w = ann['height'], ann['width']
    
        junctions = np.array(ann['junctions'], np.float32)
        # ✅ 归一化修正
        if junctions.max() <= 1.0:
            junctions[:, 0] *= w
            junctions[:, 1] *= h
        ann['junctions'] = junctions
    
        # ✅ 空负样本保护
        edges_neg = np.array(ann.get('edges_negative', []), np.int64)
        if edges_neg.size == 0:
            edges_neg = np.zeros((0, 2), dtype=np.int64)
        ann['edges_negative'] = edges_neg

        # --------------------------------------------------------
        # 🔧 自动补全 ann["lines"]（Test / Val）
        # --------------------------------------------------------
        if "lines" not in ann or len(ann["lines"]) == 0:
            lines = []
            juncs = ann["junctions"]
            edges = ann.get("edges_positive", [])
        
            for e in edges:
                j1 = juncs[e[0]]
                j2 = juncs[e[1]]
                lines.append([
                    float(j1[0]), float(j1[1]),
                    float(j2[0]), float(j2[1])
                ])
        
            ann["lines"] = lines

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    def image(self, idx):
        ann = copy.deepcopy(self.annotations[idx])
        image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        return image
    @staticmethod
    def collate_fn(batch):
        return (default_collate([b[0] for b in batch]),
                [b[1] for b in batch])

# ============================================================
# ✅ 兼容旧版接口：允许通过 "TestDataset" 工厂名称调用
# ============================================================
TestDataset = TestDatasetWithAnnotations
