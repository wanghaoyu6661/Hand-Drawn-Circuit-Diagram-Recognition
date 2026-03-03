import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = "/root/autodl-tmp/HAWP"

    
    DATASETS = {
        # -------------------------------
        # 官方 Wireframe 系列
        # -------------------------------
        'wireframe_train': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/train.json',
        },
        'wireframe_train-pseudo': {
            'img_dir': 'wireframe-pseudo/images',
            'ann_file': 'wireframe-pseudo/train.json',
        },
        'wireframe_train-syn-export': {
            'img_dir': 'wireframe-syn-export/images',
            'ann_file': 'wireframe-syn-export/train.json',
        },
        'wireframe_train-syn-export-1': {
            'img_dir': 'wireframe-syn-export-ep30-iter100-th075/images',
            'ann_file': 'wireframe-syn-export-ep30-iter100-th075/train.json',
        },
        'wireframe_test1': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/overfit.json',
        },
        # -------------------------------
        # 官方 synthetic / cities / york
        # -------------------------------
        'synthetic_train': {
            'img_dir': 'synthetic-shapes/images',
            'ann_file': 'synthetic-shapes/train.json',
        },
        'synthetic_test': {
            'img_dir': 'synthetic-shapes/images',
            'ann_file': 'synthetic-shapes/test.json',
        },
        'cities_train': {
            'img_dir': 'cities/images',
            'ann_file': 'cities/train.json',
        },
        'cities_test': {
            'img_dir': 'cities/images',
            'ann_file': 'cities/test.json',
        },
        'wireframe_test': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/test.json',
        },
        'york_test': {
            'img_dir': 'york/images',
            'ann_file': 'york/test.json',
        },
    
        # ✅ 你的电路图数据集（HAWP风格预处理后）
        'custom_train': {
            'img_dir': 'data/data_hawp_last/images',
            'ann_file': 'data/data_hawp_last/json_converted/train_hawp_style.json',
        },
        'custom_val': {
            'img_dir': 'data/data_hawp_last/images',
            'ann_file': 'data/data_hawp_last/json_converted/val_hawp_style.json',
        },

    
        # -------------------------------
        # 可选：COCO wireframe-like
        # -------------------------------
        'coco_train-val2017': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/coco-wf-val.json',
        },
        'coco_test-val2017': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/coco-wf-val.json',
        },
    }


    @staticmethod
    def get(name):
        if name not in DatasetCatalog.DATASETS:
            raise RuntimeError(f"Dataset not available: {name}")

        attrs = DatasetCatalog.DATASETS[name]
        data_dir = DatasetCatalog.DATA_DIR

        args = dict(
            root = osp.join(data_dir, attrs['img_dir']),
            ann_file = osp.join(data_dir, attrs['ann_file']),
        )

        # ✅ 明确指定类名
        if 'train' in name:
            factory = "TrainDataset"
        elif 'val' in name or 'test' in name:
            factory = "TestDataset"
        else:
            raise NotImplementedError(f"Unknown dataset type for {name}")

        return dict(factory=factory, args=args)
