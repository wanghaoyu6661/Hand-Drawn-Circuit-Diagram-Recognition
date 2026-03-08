#!/bin/bash
set -eo pipefail

###############################################
# 初始化 Conda（尽量兼容不同机器）
###############################################
CONDA_SH=""

for CAND in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh"
do
  if [ -f "$CAND" ]; then
    CONDA_SH="$CAND"
    break
  fi
done

if [ -n "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
else
  echo "❌ 找不到 conda.sh，请先安装 Conda 或手动修改脚本中的 conda.sh 路径"
  exit 1
fi

###############################################
# 项目路径 / 配置文件
###############################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$PROJECT_ROOT/src/pipeline"
DEFAULT_HAWP_CFG="$PROJECT_ROOT/configs/config.yaml"

# 优先使用本地覆盖配置（不提交 GitHub），否则用默认配置
if [ -f "$PROJECT_ROOT/configs/paths.local.yaml" ]; then
  CFG_FILE="$PROJECT_ROOT/configs/paths.local.yaml"
else
  CFG_FILE="$PROJECT_ROOT/configs/paths.yaml"
fi

# Conda 环境名可通过环境变量覆盖
CONDA_ENV_NAME="${HCD_CONDA_ENV:-hcd_pipeline_v2}"

###############################################
# 激活统一环境（只激活一次）
###############################################
set +u
conda activate "$CONDA_ENV_NAME"
set -u 2>/dev/null || true

###############################################
# 统一切到仓库根目录，避免相对路径受当前 shell 工作目录影响
###############################################
cd "$PROJECT_ROOT"

###############################################
# 第三方本地包（HAWP / PARSeq）
# 说明：fresh-machine 复现中，直接执行 python -m hawp...
# 如果没有把 third_party 目录加入 PYTHONPATH，会出现
# ModuleNotFoundError: No module named 'hawp'.
# 这里自动注入仓库内第三方路径，避免用户手动 export。
###############################################
if [ -d "$PROJECT_ROOT/third_party/HAWP" ]; then
  export PYTHONPATH="$PROJECT_ROOT/third_party/HAWP:${PYTHONPATH:-}"
fi
if [ -d "$PROJECT_ROOT/third_party/parseq-main" ]; then
  export PYTHONPATH="$PROJECT_ROOT/third_party/parseq-main:${PYTHONPATH:-}"
fi

###############################################
# YAML 读取助手
# - 统一复用 src/pipeline/config_utils.py
# - 返回的都是“已解析好的绝对路径”
###############################################
cfg_get() {
  local key="$1"
  python - "$CFG_FILE" "$key" <<'PY'
import sys
from src.pipeline.config_utils import load_paths_config

cfg_file, key = sys.argv[1], sys.argv[2]
data = load_paths_config(cfg_file)

cur = data
for p in key.split("."):
    if not isinstance(cur, dict) or p not in cur:
        print("")
        sys.exit(0)
    cur = cur[p]

print(cur if cur is not None else "")
PY
}

###############################################
# 从 YAML 读取路径（通过 config_utils 统一解析）
###############################################
# 输入 / 权重
SRC_IMG_DIR="$(cfg_get paths.src_img)"
YOLO_MODEL="$(cfg_get weights.yolo_best)"
HAWP_CFG="$(cfg_get weights.hawp_cfg)"
HAWP_CKPT="$(cfg_get weights.hawp_ckpt)"

# 输出目录
YOLO_EXP_DIR="$(cfg_get paths.yolo_detect_exp)"        # .../yolo_detect/exp
SUPPRESSED_IMG_DIR="$(cfg_get paths.suppressed_img)"   # .../suppressed_img
HAWP_JSON_DIR="$(cfg_get paths.hawp_json)"             # .../HAWPimg/json
FONT_DIR="$(cfg_get paths.font_dir)"                   # .../assets/fonts
SPICE_NETLIST_DIR="$(cfg_get paths.spice_netlists)"    # .../spice_netlists

# 派生目录
YOLO_PROJECT_DIR="$(dirname "$YOLO_EXP_DIR")"          # .../yolo_detect
YOLO_NAME="$(basename "$YOLO_EXP_DIR")"                # exp
HAWP_OUT_DIR="$(dirname "$HAWP_JSON_DIR")"             # .../HAWPimg
RUN_DIR="$(dirname "$(dirname "$YOLO_EXP_DIR")")"      # .../run1（假设 yolo_detect/exp 结构）

###############################################
# 路径检查（尽早失败，便于定位）
###############################################
need_nonempty() {
  local name="$1" val="$2"
  if [ -z "$val" ]; then
    echo "❌ 配置缺失: $name（检查 $CFG_FILE）"
    exit 1
  fi
}
need_file() {
  local name="$1" p="$2"
  if [ ! -f "$p" ]; then
    echo "❌ 文件不存在: $name -> $p"
    exit 1
  fi
}
need_dir() {
  local name="$1" p="$2"
  if [ ! -d "$p" ]; then
    echo "❌ 目录不存在: $name -> $p"
    exit 1
  fi
}

need_nonempty "paths.src_img" "$SRC_IMG_DIR"
need_nonempty "weights.yolo_best" "$YOLO_MODEL"
need_nonempty "weights.hawp_cfg" "$HAWP_CFG"
need_nonempty "weights.hawp_ckpt" "$HAWP_CKPT"
need_nonempty "paths.yolo_detect_exp" "$YOLO_EXP_DIR"
need_nonempty "paths.suppressed_img" "$SUPPRESSED_IMG_DIR"
need_nonempty "paths.hawp_json" "$HAWP_JSON_DIR"
need_nonempty "paths.font_dir" "$FONT_DIR"
need_nonempty "paths.spice_netlists" "$SPICE_NETLIST_DIR"

need_dir  "paths.src_img" "$SRC_IMG_DIR"
need_file "weights.yolo_best" "$YOLO_MODEL"
need_file "weights.hawp_cfg" "$HAWP_CFG"
need_file "weights.hawp_ckpt" "$HAWP_CKPT"

###############################################
# 目录准备
###############################################
mkdir -p "$RUN_DIR" "$YOLO_PROJECT_DIR" "$HAWP_OUT_DIR" "$FONT_DIR" "$SPICE_NETLIST_DIR"

# 字体准备（避免 build_final_json 卡住）
if [ ! -f "$FONT_DIR/DejaVuSans.ttf" ] && [ -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ]; then
  cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf "$FONT_DIR/" 2>/dev/null || true
fi

echo "======================================="
echo "🔥 全流程开始运行（YAML驱动版）"
echo "📁 PROJECT_ROOT   = $PROJECT_ROOT"
echo "📄 CFG_FILE       = $CFG_FILE"
echo "🐍 CONDA_ENV      = $CONDA_ENV_NAME"
echo "🧩 HAWP_CFG       = $HAWP_CFG"
echo "📁 RUN_DIR        = $RUN_DIR"
echo "📁 SPICE_OUT_DIR  = $SPICE_NETLIST_DIR"
echo "🧭 PYTHONPATH     = $PYTHONPATH"
echo "======================================="

if [ "$HAWP_CFG" = "$DEFAULT_HAWP_CFG" ]; then
  echo "ℹ️ HAWP 配置默认来自: $DEFAULT_HAWP_CFG"
fi

python - <<'PY'
import sys
failed = []
for name in ["hawp", "strhub"]:
    try:
        __import__(name)
    except Exception as e:
        failed.append((name, repr(e)))
if failed:
    print("❌ 第三方本地包导入失败。请确认 third_party 目录完整，或手动执行:")
    print("   pip install -e third_party/HAWP")
    print("   pip install -e third_party/parseq-main")
    for name, err in failed:
        print(f"   - {name}: {err}")
    sys.exit(1)
print("✔ 第三方本地包导入检查通过（hawp / strhub）")
PY

############################################
# 1. YOLOv10 检测
############################################
echo ""
echo "🚀 [1/9] YOLOv10 检测开始..."

yolo detect predict \
    model="$YOLO_MODEL" \
    source="$SRC_IMG_DIR" \
    conf=0.45 \
    iou=0.40 \
    save_txt=True \
    save_conf=True \
    save_crop=True \
    project="$YOLO_PROJECT_DIR" \
    name="$YOLO_NAME" \
    exist_ok=True

python "$PIPELINE_DIR/scanify_folder.py"
python "$PIPELINE_DIR/make_yolov10_crops_by_image.py"

echo "✔ YOLOv10 完成"
echo "---------------------------------------"

############################################
# 2. PARSeq 文本识别
############################################
echo "🚀 [2/9] PARSeq 开始..."
python "$PIPELINE_DIR/fuse_yolo_parseq.py"
echo "✔ PARSeq 完成"
echo "---------------------------------------"

############################################
# 3. 元件区域抑制（用于连接推理）
############################################
echo "🚀 [3/9] 元件区域抑制开始..."
python "$PIPELINE_DIR/suppress_component_regions.py"
echo "✔ 元件区域抑制完成"
echo "---------------------------------------"

############################################
# 4. HAWP 点检测（外部 repo）
############################################
echo "🚀 [4/9] HAWP 开始..."

python -m hawp.fsl.predict_circuit_batch_fixed \
   --config "$HAWP_CFG" \
   --ckpt "$HAWP_CKPT" \
   --input-dir "$SUPPRESSED_IMG_DIR" \
   --output-dir "$HAWP_OUT_DIR" \
   --junc-th 0.25

echo "✔ HAWP 完成"
echo "---------------------------------------"

############################################
# 5. 合并点
############################################
echo "🚀 [5/9] merge_points.py 开始..."
python "$PIPELINE_DIR/merge_points.py"
echo "✔ merge_points 完成"
echo "---------------------------------------"

############################################
# 6. 去组件 / 导线增强
############################################
echo "🚀 [6/9] remove_components.py 开始..."
python "$PIPELINE_DIR/remove_components.py"
echo "✔ remove_components 完成"
echo "---------------------------------------"

############################################
# 7. 生成连接 + 类型细化 + 端点语义
############################################
echo "🚀 [7/9] build_connections.py 开始..."
python "$PIPELINE_DIR/build_connections.py"
python "$PIPELINE_DIR/refine_component_types.py"
python "$PIPELINE_DIR/infer_port_vitpose.py"
echo "✔ build_connections / refine / port inference 完成"
echo "---------------------------------------"

############################################
# 8. 生成最终 JSON + 可视化
############################################
echo "🚀 [8/9] build_final_json.py 开始..."
python "$PIPELINE_DIR/build_final_json.py"
echo "✔ build_final_json 完成"
echo "---------------------------------------"

############################################
# 9. 将 final JSON 转换为 SPICE 网表
############################################
echo "🚀 [9/9] build_spice_netlists.py 开始..."
python "$PIPELINE_DIR/build_spice_netlists.py" --cfg "$CFG_FILE" --output-dir "$SPICE_NETLIST_DIR"
echo "✔ build_spice_netlists 完成"
echo "---------------------------------------"

echo "======================================="
echo "🎉 全流程执行完成（YAML驱动版，含 SPICE 网表导出）"
echo "======================================="
