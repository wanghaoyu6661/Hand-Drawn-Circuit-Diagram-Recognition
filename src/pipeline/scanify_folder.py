import os
from glob import glob
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from path_config import cfg_get, project_path

IN_DIR  = cfg_get("paths", "src_img", default=project_path("data", "inputs"))
OUT_DIR = cfg_get("paths", "src_img_scanned", default=project_path("outputs", "run1", "src_img_scanned"))

os.makedirs(OUT_DIR, exist_ok=True)

def scanify(im: Image.Image) -> Image.Image:
    # 1) 保留 RGB，不直接灰度化
    im = im.convert("RGB")

    # 2) 用灰度估计背景光照（只用于计算光照场）
    g = ImageOps.grayscale(im)
    bg = g.filter(ImageFilter.GaussianBlur(radius=25))

    # 3) 计算光照归一化因子：scale = 255 / bg
    g_px = g.load()
    bg_px = bg.load()
    w, h = g.size

    # 4) 对 RGB 每个通道做同一个光照归一化（保留颜色比例）
    r, gch, b = im.split()
    rp, gp, bp = r.load(), gch.load(), b.load()

    out = Image.new("RGB", (w, h), (255, 255, 255))
    outp = out.load()

    for y in range(h):
        for x in range(w):
            bgl = bg_px[x, y]
            if bgl < 1:
                bgl = 1
            scale = 255.0 / float(bgl)

            rr = int(min(255, rp[x, y] * scale))
            gg = int(min(255, gp[x, y] * scale))
            bb = int(min(255, bp[x, y] * scale))
            outp[x, y] = (rr, gg, bb)

    # 5) 自动对比 + 轻微对比增强（别太狠）
    out = ImageOps.autocontrast(out, cutoff=1)
    out = ImageEnhance.Contrast(out).enhance(1.15)

    # 6) 轻微去饱和（模拟扫描件，但不要变成纯黑）
    out = ImageEnhance.Color(out).enhance(0.85)

    return out

imgs = []
for ext in ("*.jpg","*.png","*.jpeg","*.JPG","*.PNG","*.JPEG"):
    imgs += glob(os.path.join(IN_DIR, ext))

imgs = sorted(imgs)
print(f"[SCANIFY] found {len(imgs)} images in {IN_DIR}")

for p in imgs:
    bn = os.path.basename(p)
    im = Image.open(p).convert("RGB")
    out = scanify(im)
    out.save(os.path.join(OUT_DIR, bn), quality=95)
    print("[OK]", bn)

print("[DONE] ->", OUT_DIR)
