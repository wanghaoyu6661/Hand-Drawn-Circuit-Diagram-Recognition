from . import models

def build_model(cfg):
    model = models.WireframeDetector(cfg)

    # ✅ 手动补充重要参数传入模型属性
    print("debug_cfg_after_WireframeDetector._init_---------------------------------------------")
    print(type(cfg))
    print(cfg)
    print("debug_cfg_after_WireframeDetector._init_---------------------------------------------")
    ph = getattr(cfg, "PARSING_HEAD", None)
    print("debug_ph-----------------------------------------------------")
    print(ph)
    print("debug_ph-----------------------------------------------------")
    if ph is not None:
        # J2L_THRESHOLD → 半径（像素）
        if hasattr(ph, "J2L_THRESHOLD"):
            model.j2l_radius_px = float(ph.J2L_THRESHOLD)
            print(f"[config] ✅ Loaded J2L_THRESHOLD = {model.j2l_radius_px}")

        # 其余可选参数也可以一并带入模型
        for key in ["JMATCH_THRESHOLD", "MAX_DISTANCE", "N_DYN_JUNC"]:
            if hasattr(ph, key):
                val = getattr(ph, key)
                setattr(model, key.lower(), val)
                print(f"[config] set {key.lower()} = {val}")

    return model
