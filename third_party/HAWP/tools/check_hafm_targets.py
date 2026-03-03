# tools/check_hafm_targets.py
import os, json, random, argparse
import numpy as np
import cv2

def load_ann(ann_file):
    with open(ann_file, 'r') as f:
        return json.load(f)

def draw_points(img, pts, color=(0,0,255), r=4):
    out = img.copy()
    for x, y in pts:
        cv2.circle(out, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--num', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target-w', type=int, default=128)
    ap.add_argument('--target-h', type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    anns = load_ann(args.ann_file)
    random.seed(args.seed)
    random.shuffle(anns)
    picked = anns[:args.num]

    for ann in picked:
        fn = ann['filename']
        w, h = int(ann['width']), int(ann['height'])
        img_path = os.path.join(args.img_root, fn)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'[skip] cannot read {img_path}')
            continue

        j = np.array(ann.get('junctions', []), np.float32)
        if j.size == 0:
            print(f'[warn] no junctions in {fn}')
            continue

        # 0~1 -> 像素
        if j.max() <= 1.0:
            j[:, 0] *= w
            j[:, 1] *= h

        # 图1：原图叠加
        vis1 = draw_points(img, j, (0,0,255), 4)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(fn).rsplit('.',1)[0] + '_junctions_on_image.jpg'), vis1)

        # 图2：统一缩放到 target 尺寸后的叠加
        sx = args.target_w / float(w)
        sy = args.target_h / float(h)
        j_t = j.copy()
        j_t[:, 0] *= sx
        j_t[:, 1] *= sy
        canvas = np.zeros((args.target_h, args.target_w, 3), np.uint8)
        vis2 = draw_points(canvas, j_t, (0,0,255), 3)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(fn).rsplit('.',1)[0] + '_target_size_overlay.jpg'), vis2)

        print(f'[ok] {fn} -> {vis1.shape} / target ({args.target_w}x{args.target_h})')

if __name__ == '__main__':
    main()
