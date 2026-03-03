#!/bin/bash

###############################################
# 初始化 Conda（必须有，否则 activate 无效）
###############################################

# AutoDL 默认 Miniconda 路径
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
fi

# 若用户有自己的 bashrc，也一起加载
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

echo "======================================="
echo "🔥 全流程开始运行：YOLO → PARSeq → LaMa → HAWP → merge → remove → link → final_json"
echo "======================================="


############################################
# 1. YOLOv9 检测
############################################
echo ""
echo "🚀 [1/8] YOLOv10 检测开始..."

conda activate yolov10

yolo detect predict \
    model=/root/runs/detect/train4/weights/best.pt \
    source=/root/autodl-tmp/final_result/src_img \
    conf=0.45 \
    iou=0.40 \
    save_txt=True \
    save_conf=True \
    save_crop=True \
    project=/root/autodl-tmp/final_result/yolo_detect \
    name=exp \
    exist_ok=True

python /root/autodl-tmp/final_result/Integrate_code/scanify_folder.py
python /root/autodl-tmp/final_result/Integrate_code/make_yolov10_crops_by_image.py

echo "✔ YOLOv10 完成"
echo "---------------------------------------"


############################################
# 2. PARSeq 文本识别
############################################
echo "🚀 [2/8] PARSeq 开始..."

conda activate parseq_env

python /root/autodl-tmp/final_result/Integrate_code/fuse_yolo_parseq.py

echo "✔ PARSeq 完成"
echo "---------------------------------------"


############################################
# 3. LaMa 擦除元件
############################################
echo "🚀 [3/8] LaMa 开始..."

conda activate lama_env

python /root/autodl-tmp/lama_prj/clean_with_yolov9_lama.py

echo "✔ LaMa 完成"
echo "---------------------------------------"


############################################
# 4. HAWP 点检测
############################################
echo "🚀 [4/8] HAWP 开始..."

conda activate hawp

python -m hawp.fsl.predict_circuit_batch_fixed \
   --config /root/autodl-tmp/output_hawp_last/hawpv2/251231-182855/config.yaml \
   --ckpt /root/autodl-tmp/output_hawp_last/hawpv2/251231-182855/last_epochs/last_epoch_035.pth \
   --input-dir /root/autodl-tmp/final_result/lama_clean \
   --output-dir /root/autodl-tmp/final_result/HAWPimg \
   --junc-th 0.25

echo "✔ HAWP 完成"
echo "---------------------------------------"


############################################
# 5. 合并点
############################################
echo "🚀 [5/8] merge_points.py 开始..."

python /root/autodl-tmp/final_result/Integrate_code/merge_points.py

echo "✔ merge_points 完成"
echo "---------------------------------------"


############################################
# 6. 去组件 remove_components
############################################
echo "🚀 [6/8] remove_components.py 开始..."

python /root/autodl-tmp/final_result/Integrate_code/remove_components.py

echo "✔ remove_components 完成"
echo "---------------------------------------"


############################################
# 7. 生成连接
############################################
echo "🚀 [7/8] build_connections.py 开始..."

python /root/autodl-tmp/final_result/Integrate_code/build_connections.py

conda activate dinov2_cls
python /root/autodl-tmp/final_result/Integrate_code/refine_component_types.py

conda activate vitpose_clean
python /root/autodl-tmp/final_result/Integrate_code/infer_port_vitpose.py

echo "✔ build_connections 完成"
echo "---------------------------------------"


############################################
# 8. 生成最终 JSON + 可视化
############################################
echo "🚀 [8/8] build_final_json.py 开始..."

python /root/autodl-tmp/final_result/Integrate_code/build_final_json.py

echo "✔ build_final_json 完成"
echo "---------------------------------------"

echo "======================================="
echo "🎉 全流程执行完成，所有结果已生成"
echo "======================================="
