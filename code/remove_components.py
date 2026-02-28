#!/usr/bin/env python3
# autodl-tmp/Integrate_code/remove_components.py
import cv2
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
SRC_IMG_DIR   = Path(r'/root/autodl-tmp/final_result/src_img')          # 原始 png
YOLO_TXT_DIR  = Path(r'/root/autodl-tmp/final_result/yolo_detect/exp/labels')  # yolo txt
SAVE_DIR      = Path(r'/root/autodl-tmp/final_result/dao_xian')  # 输出目录
SAVE_DIR.mkdir(exist_ok=True)

CLASSES_TXT   = Path(r'/root/autodl-fs/kicad_component_dataset/classes.txt')  # 可选，仅用于打印
CLASS_NAMES   = [l.strip() for l in open(CLASSES_TXT).readlines()] if CLASSES_TXT.exists() else []

# ------------------------------------------------------------------
# 超参数 - 修改这里来调整框的大小和噪声过滤
# ------------------------------------------------------------------
EXPAND_PIXELS_INTEGRATED_CIRCUIT = 0   # 框向外扩展的像素数，正数表示扩大，负数表示缩小
EXPAND_PIXELS_OTHER = 0

# 噪声过滤参数
NOISE_FILTER_ENABLED = True  # 建议打开，否则下面参数都不生效

# ---- 自适应开关（推荐 True）----
NOISE_FILTER_ADAPTIVE = True

# ---- 自适应系数（按 min(H,W) 比例）----
MIN_WIRE_LENGTH_SCALE = 0.006   # ⭐ 最关键：0.005~0.008 自己调
MIN_WIRE_AREA_SCALE   = 0.004   # 次关键：0.003~0.006 自己调

# ---- 自适应 clamp（防止小图过小、大图过大）----
MIN_WIRE_LENGTH_MIN, MIN_WIRE_LENGTH_MAX = 15, 80
MIN_WIRE_AREA_MIN,   MIN_WIRE_AREA_MAX   = 20, 150

# ---- 非自适应时的固定默认值 ----
MIN_WIRE_AREA   = 30
MIN_WIRE_LENGTH = 25

# 形态学开运算：默认关闭（=1），不建议自适应
MORPH_OPEN_KS = 1

# 导线修复参数
WIRE_REPAIR_ENABLED = True   # 是否启用导线修复
DILATION_KERNEL_SIZE = 0     # 膨胀核大小

# 超大图加速：先缩小做噪声过滤，再放大回原图
NOISE_FILTER_DOWNSCALE = True
NOISE_FILTER_MAX_PIXELS = 2_000_000  
# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def yolo2xyxy(img_h, img_w, x_center, y_center, w, h):
    """YOLO 归一化 xywh -> 左上角+右下角 (像素)"""
    x1 = int((x_center - w/2) * img_w)
    y1 = int((y_center - h/2) * img_h)
    x2 = int((x_center + w/2) * img_w)
    y2 = int((y_center + h/2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def expand_bbox(x1, y1, x2, y2, expand_pixels, img_w, img_h):
    """扩展边界框"""
    x1_expanded = max(0, x1 - expand_pixels)
    y1_expanded = max(0, y1 - expand_pixels)
    x2_expanded = min(img_w, x2 + expand_pixels)
    y2_expanded = min(img_h, y2 + expand_pixels)
    return x1_expanded, y1_expanded, x2_expanded, y2_expanded

def filter_noise_wires(bin_img, min_area=50, min_length=30, morph_ks=3,
                       enable_downscale=True, max_pixels=6_000_000):
    """
    超快版噪声过滤（解决大图 connected-components + per-component contours 极慢的问题）

    核心优化：
    - 不再对每个连通域做 component_mask/findContours/minAreaRect（那会对大图爆炸）
    - 直接使用 connectedComponentsWithStats 的 stats（area/width/height）
    - 用 keep[labels] 一次性生成保留掩膜（向量化）

    可选优化：
    - 超大图先 downscale 做过滤，再 nearest upsample 回原图与原图 AND（更省时）
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img

    H, W = bin_img.shape[:2]

    # --- 可选：超大图先缩小做噪声过滤 ---
    if enable_downscale and (H * W > max_pixels):
        scale = (max_pixels / float(H * W)) ** 0.5
        newW = max(64, int(round(W * scale)))
        newH = max(64, int(round(H * scale)))
        small = cv2.resize(bin_img, (newW, newH), interpolation=cv2.INTER_NEAREST)

        filtered_small = filter_noise_wires(
            small,
            min_area=max(1, int(round(min_area * scale * scale))),   # 面积按比例缩放
            min_length=max(1, int(round(min_length * scale))),       # 长度按比例缩放
            morph_ks=max(1, int(round(morph_ks * scale))),
            enable_downscale=False,  # 防止递归再缩小
            max_pixels=max_pixels
        )

        up = cv2.resize(filtered_small, (W, H), interpolation=cv2.INTER_NEAREST)
        # 只删除在小图中被判定为噪声的区域
        out = cv2.bitwise_and(bin_img, up)
        return out

    # 保证是二值 0/255
    filtered_img = (bin_img > 0).astype(np.uint8) * 255

    # 1) 形态学开运算（可选）
    if morph_ks and morph_ks > 1:
        kernel = np.ones((morph_ks, morph_ks), np.uint8)
        filtered_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, kernel)
        print(f'[NOISE_FILTER] Applied morphological opening with kernel {morph_ks}x{morph_ks}')

    # 2) 连通域统计
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_img, connectivity=8)
    total_components = num_labels - 1

    if total_components <= 0:
        print('[NOISE_FILTER] No components found')
        return filtered_img

    # stats: [label, 5] => [x, y, w, h, area]
    areas = stats[:, cv2.CC_STAT_AREA]
    ws    = stats[:, cv2.CC_STAT_WIDTH]
    hs    = stats[:, cv2.CC_STAT_HEIGHT]
    lengths = np.maximum(ws, hs)

    # 3) 构建 keep 表（0 是背景必须保留）
    keep = np.ones((num_labels,), dtype=np.bool_)
    keep[0] = True
    keep[1:] = (areas[1:] >= int(min_area)) & (lengths[1:] >= int(min_length))

    noise_count = int(np.sum(~keep[1:]))

    # 4) 向量化生成掩膜并应用（一次性全图）
    mask_keep = keep[labels]  # shape: HxW (bool)
    filtered_img = cv2.bitwise_and(filtered_img, filtered_img, mask=mask_keep.astype(np.uint8) * 255)

    print(f'[NOISE_FILTER] Removed {noise_count}/{total_components} noise components')
    print(f'[NOISE_FILTER] Remaining components: {total_components - noise_count}')

    return filtered_img


def _line_kernel(length: int, angle: int) -> np.ndarray:
    """
    生成细长线形结构元（1像素厚），angle ∈ {0,45,90,135}
    length 必须 >= 3 且为奇数更对称
    """
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1

    k = np.zeros((length, length), np.uint8)
    c = length // 2

    if angle == 0:
        k[c, :] = 1
    elif angle == 90:
        k[:, c] = 1
    elif angle == 45:
        # 左下到右上
        for i in range(length):
            k[length - 1 - i, i] = 1
    elif angle == 135:
        # 左上到右下
        for i in range(length):
            k[i, i] = 1
    else:
        raise ValueError("angle must be one of {0,45,90,135}")
    return k


def repair_broken_wires(
    bin_img,
    dilation_iterations=1,      # 兼容旧参数；此方案里不太需要多次迭代
    kernel_size=0,              # <=0 表示自动
    auto_scale=0.012,           # 建议：min(H,W)*0.8% 作为线形 kernel 长度
    k_min=3,
    k_max=35,
    do_final_thin=False,        # 可选：如果你有 ximgproc.thinning
):
    """
    更稳的导线修复：
    - 使用 0/45/90/135° 四方向的线形 kernel 做 MORPH_CLOSE 补断线
    - kernel_size<=0 时按图像尺寸自适应
    """
    if bin_img is None:
        return bin_img

    # 保证二值 0/255
    img = (bin_img > 0).astype(np.uint8) * 255

    H, W = img.shape[:2]
    min_hw = min(H, W)

    if kernel_size is None or kernel_size <= 0:
        # 自适应长度（可按你数据再调 auto_scale）
        klen = int(round(min_hw * float(auto_scale)))
        klen = max(int(k_min), min(int(klen), int(k_max)))
    else:
        klen = int(kernel_size)
        klen = max(int(k_min), min(int(klen), int(k_max)))

    # 4 个方向做 close（补断线更稳：不会无限变粗）
    angles = [0, 45, 90, 135]
    out = np.zeros_like(img)

    for ang in angles:
        k = _line_kernel(klen, ang)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
        out = cv2.bitwise_or(out, closed)

    # 可选：轻微去毛刺（防止 close 带来的小鼓包）
    # 注意：别开太大，否则会吃掉细线
    # out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    # 可选：如果你环境里有 ximgproc，可以做一次细化让线变回细线
    if do_final_thin:
        try:
            out = cv2.ximgproc.thinning(out)
        except Exception:
            pass

    print(f"[WIRE_REPAIR] directional-close klen={klen} (auto={kernel_size<=0}), angles={angles}")
    return out


def calculate_wire_statistics(bin_img):
    """计算导线统计信息"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    
    if num_labels <= 1:
        return {"total_components": 0, "areas": [], "avg_area": 0}
    
    areas = []
    for i in range(1, num_labels):  # 跳过背景
        areas.append(stats[i, cv2.CC_STAT_AREA])
    
    return {
        "total_components": num_labels - 1,
        "areas": areas,
        "avg_area": np.mean(areas) if areas else 0,
        "max_area": max(areas) if areas else 0,
        "min_area": min(areas) if areas else 0
    }

def process_one_image(stem: str):
    # 自动匹配真正存在的图片文件
    candidates = list(SRC_IMG_DIR.glob(f"{stem}.*"))
    if not candidates:
        print(f'[WARN] skip {stem} (not exist)')
        return
    
    img_path = candidates[0]   # 真实文件路径，例如 .jpg / .png / .jpeg
    txt_path = YOLO_TXT_DIR / f'{stem}.txt'

    # --- 以下全部保持你原来的代码 ---


    if not img_path.exists():
        print(f'[WARN] skip {img_path} (not exist)')
        return
    
    def clamp(v, vmin, vmax):
        return max(vmin, min(v, vmax))
    
    # 读取图像
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[:2]
    # ===== 自适应噪声过滤阈值（每张图单独算）=====
    if NOISE_FILTER_ADAPTIVE:
        base = min(H, W)
        MIN_WIRE_LENGTH_CUR = int(clamp(base * MIN_WIRE_LENGTH_SCALE,
                                        MIN_WIRE_LENGTH_MIN, MIN_WIRE_LENGTH_MAX))
        MIN_WIRE_AREA_CUR   = int(clamp(base * MIN_WIRE_AREA_SCALE,
                                        MIN_WIRE_AREA_MIN, MIN_WIRE_AREA_MAX))
    else:
        MIN_WIRE_LENGTH_CUR = int(MIN_WIRE_LENGTH)
        MIN_WIRE_AREA_CUR   = int(MIN_WIRE_AREA)

    print(f"[CFG@{stem}] adaptive={NOISE_FILTER_ADAPTIVE} "
        f"MIN_WIRE_LENGTH={MIN_WIRE_LENGTH_CUR}, MIN_WIRE_AREA={MIN_WIRE_AREA_CUR}, MORPH_OPEN_KS={MORPH_OPEN_KS}")
    min_hw = min(H, W)

    # 1) 轻微去噪，压纸纹/压缩噪声（不会明显吃线）
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) 背景校正（关键！解决阴影/亮度不均导致的断线和白噪点）
    # sigma 随分辨率变化：图越大，背景变化越缓，用更大的 sigma
    sigma = max(15, int(round(min_hw / 30)))     # 你可调：/25 更强校正，/40 更弱
    bg = cv2.GaussianBlur(gray_blur, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 用除法做“平场校正”：把背景影响压掉，让黑线在全图都“同一种黑”
    norm = cv2.divide(gray_blur, bg, scale=255)

    # 1. Otsu（干净）
    _, bin_otsu = cv2.threshold(
        norm, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Adaptive（高召回，捞浅线）—— blockSize 自适应
    # 经验：blockSize ~ 1.5% * min(H,W)，并且必须是奇数且 >= 15
    bs = int(round(min_hw * 0.015))          # 4000→60；1000→15
    bs = max(15, min(bs, 81))                # clamp，避免过小/过大
    if bs % 2 == 0:
        bs += 1

    # C 也可以轻微随图变，但先保持你原来的 5（后续再调）
    C_adp = 5

    bin_adp = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=bs,
        C=C_adp
    )

    # 只在“较暗区域”使用 adaptive 的结果，避免背景被捞白
    # 改成：按 norm 的分位数自适应阈值（每张图自适应亮度/纸纹）
    P_DARK = 30   # 你可调：20(更干净) ~ 40(更保守/更高召回)
    thr = int(np.percentile(norm, P_DARK))

    # 再做一个 clamp，防止极端图把阈值拉太离谱
    thr = max(190, min(thr, 240))

    dark_mask = (norm < thr).astype(np.uint8) * 255
    print(f"[DBG@{stem}] min_hw={min_hw} bs={bs} C={C_adp} P_DARK={P_DARK} thr={thr}")
    bin_img = cv2.bitwise_or(bin_otsu, cv2.bitwise_and(bin_adp, dark_mask))

    # 4) 轻度闭运算：专门补“断断续续”的线段（比开运算更适合你的目标）
    # 核大小随分辨率走：大图用稍大核，小图用小核
    ks = 3 if min_hw < 1200 else 5 if min_hw < 2600 else 7
    kernel = np.ones((ks, ks), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    H, W = bin_img.shape
    print(f'[INFO] Processing {stem}, image size: {W}x{H}')

    # 创建调试图像（可选）
    debug_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 读取 yolo txt
    if txt_path.exists():
        with open(txt_path) as f:
            lines = f.readlines()
    else:
        lines = []

    boxes_count = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        
        cls_id, x_c, y_c, w, h, conf = map(float, parts)
        x1, y1, x2, y2 = yolo2xyxy(H, W, x_c, y_c, w, h)
        
        if int(cls_id) == 1:   # junction
            # junction 不扩展，不涂灰
            print(f"[SKIP] skip junction bbox for class 1 at {stem}")
            continue
        # 扩展边界框
        if EXPAND_PIXELS_INTEGRATED_CIRCUIT != 0 and cls_id in [33,34,35]:
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, EXPAND_PIXELS_INTEGRATED_CIRCUIT, W, H)
            
        if EXPAND_PIXELS_OTHER != 0 and cls_id not in [33,34,35]:
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, EXPAND_PIXELS_OTHER, W, H)
        
        # 在调试图像上画原始框（绿色）和扩展框（红色）
        cv2.rectangle(debug_img, 
                     (int((x_c - w/2) * W), int((y_c - h/2) * H)), 
                     (int((x_c + w/2) * W), int((y_c + h/2) * H)), 
                     (0, 255, 0), 1)  # 原始框 - 绿色
        
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 扩展框 - 红色
        
        # 把扩展后的框内像素置为背景色
        cv2.rectangle(bin_img, (x1, y1), (x2, y2), 0, -1)
        boxes_count += 1
        
        # 打印框信息
        if CLASS_NAMES and int(cls_id) < len(CLASS_NAMES):
            class_name = CLASS_NAMES[int(cls_id)]

    
    # 保存仅移除元件的结果（不含噪声过滤）
    removal_only_path = SAVE_DIR / f'{stem}_boxes_expanded.png'
    cv2.imwrite(str(removal_only_path), bin_img)
    
    # 噪声过滤
    if NOISE_FILTER_ENABLED:
        
        # 保存元件移除后的状态用于过滤
        after_removal_img = bin_img.copy()
        
        # 过滤前的统计（基于元件移除后的图像）
        stats_before = calculate_wire_statistics(after_removal_img)
        print(f'[STATS] Before filtering: {stats_before["total_components"]} components, '
              f'avg area: {stats_before["avg_area"]:.1f}')
        
        bin_img_filtered = filter_noise_wires(
            after_removal_img,
            min_area=MIN_WIRE_AREA_CUR,
            min_length=MIN_WIRE_LENGTH_CUR,
            morph_ks=MORPH_OPEN_KS,
            enable_downscale=NOISE_FILTER_DOWNSCALE,
            max_pixels=NOISE_FILTER_MAX_PIXELS
        )

        # 过滤后的统计
        stats_after = calculate_wire_statistics(bin_img_filtered)
        print(f'[STATS] After filtering: {stats_after["total_components"]} components, '
              f'avg area: {stats_after["avg_area"]:.1f}')
        
        # 保存噪声过滤后的结果
        noise_filtered_path = SAVE_DIR / f'{stem}_noise_filtered.png'
        cv2.imwrite(str(noise_filtered_path), bin_img_filtered)
        print(f'[DEBUG] Noise filtered image -> {noise_filtered_path}')
        
        # 更新当前图像为噪声过滤后的结果
        current_img = bin_img_filtered
    else:
        # 如果不启用噪声过滤，当前图像就是元件移除后的图像
        current_img = bin_img

    # 导线修复 - 在噪声过滤后执行
    if WIRE_REPAIR_ENABLED:
        print(f'[INFO] Applying wire repair...')
        
        # 修复前的图像状态
        before_repair_img = current_img.copy()
        
        # 应用导线修复
        repaired_img = repair_broken_wires(
            before_repair_img,
            kernel_size=DILATION_KERNEL_SIZE
        )
        
        # 更新最终图像为修复后的结果
        final_img = repaired_img
    else:
        # 如果不启用导线修复，最终图像就是当前图像
        final_img = current_img

    # 保存最终结果
    final_path = SAVE_DIR / f'{stem}_final.png'
    cv2.imwrite(str(final_path), final_img)
    print(f'[OK] Final result saved -> {final_path}')
    
    # 保存调试图像
    debug_save_path = SAVE_DIR / f'{stem}_debug_boxes.png'
    cv2.imwrite(str(debug_save_path), debug_img)
    print(f'[DEBUG] Box visualization -> {debug_save_path}')

# ------------------------------------------------------------------
# 批量处理函数
# ------------------------------------------------------------------
def process_all_images():
    """处理 SRC_IMG_DIR 中所有图像文件（png/jpg/jpeg/bmp/tif）"""
    
    img_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
        img_files += list(SRC_IMG_DIR.glob(ext))
    
    print(f'[INFO] Found {len(img_files)} image files to process')
    
    for img_path in img_files:
        stem = img_path.stem  # 不含扩展名
        process_one_image(stem)



# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
if __name__ == '__main__':
    print(f'[CONFIG] Source directory: {SRC_IMG_DIR}')
    print(f'[CONFIG] YOLO labels directory: {YOLO_TXT_DIR}')
    print(f'[CONFIG] Save directory: {SAVE_DIR}')
    
    # 处理所有图像
    process_all_images()
    print('[DONE] All images processed')