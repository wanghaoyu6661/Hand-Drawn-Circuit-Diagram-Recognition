# Hand-Drawn-Circuit-Diagram-Recognition

Deep learning–based recognition and structural parsing pipeline for **hand-drawn circuit diagrams**.

This project provides an **end-to-end framework** that converts hand-drawn circuit schematics into structured machine-readable representations, including:

- intermediate outputs for each processing stage
- final structured JSON (`*.final.json`)
- visualization images (`*_final.png`)
- SPICE netlists (`*.spice`) ready for circuit simulation tools

The system combines deep learning perception modules with structural reasoning to recover circuit topology, component semantics, and simulation-oriented structure from noisy hand-drawn inputs.

## Project Resources

- **GitHub repository (code):** `https://github.com/wanghaoyu6661/Hand-Drawn-Circuit-Diagram-Recognition`
- **Hugging Face repository (model weights):** `https://huggingface.co/why0722/hcd-circuit-weights`

The GitHub repository contains the codebase, scripts, configs, and documentation. The Hugging Face repository hosts the released pretrained weights required for public reproduction.

---

## 1. Repository Structure

Typical repository layout:

```text
Hand-Drawn-Circuit-Diagram-Recognition/
│
├── assets/                     # static resources (fonts, metadata)
├── configs/                    # portable YAML configuration files
│   ├── config.yaml             # HAWP config used by the pipeline
│   ├── paths.yaml
│   ├── paths.example.yaml
│   └── mmpose_ports/
│
├── data/
│   ├── inputs/                 # input circuit images (includes 5 example hand-drawn circuits)
│   └── ground_truth/           # corresponding GT annotations for the 5 example circuits
│
├── docs/                       # supplementary docs / notes
├── logs/                       # runtime logs
├── outputs/                    # pipeline outputs
├── reports/
│   └── eval/                   # evaluation reports
│
├── scripts/
│   ├── run_all.sh              # full pipeline entry point
│   └── eval/                   # evaluation scripts
│
├── src/
│   └── pipeline/
│       ├── config_utils.py     # shared portable config loader
│       ├── path_config.py      # compatibility wrapper
│       ├── fuse_yolo_parseq.py
│       ├── build_connections.py
│       ├── build_final_json.py
│       ├── build_spice_netlists.py
│       └── ...
│
├── third_party/                # external repositories (HAWP, PARSeq, MMPose)
├── tests/                      # tests / local checks
├── weights/                    # pretrained model weights
├── environment.yml             # recommended public reproduction environment
├── environment.full.yml        # stricter pinned reproduction environment
└── README.md
```

### Important note on `third_party/mmpose`

`third_party/mmpose` is tracked as a **Git submodule** pinned to a specific upstream commit. That means it behaves differently from ordinary folders such as `third_party/HAWP` or `third_party/parseq-main`.

If you download this project as a **GitHub zip/tar source archive**, or use a clone that does **not** initialize submodules, the `third_party/mmpose` directory may be incomplete or may only contain a submodule pointer. In that case, MMPose config files such as:

```text
third_party/mmpose/configs/_base_/default_runtime.py
```

will be missing, and the pipeline will fail during the ViTPose stage.

For this reason, the recommended way to obtain the repository is a recursive clone.

---

## 2. Clone the Repository

Recommended:

```bash
git clone --recursive https://github.com/wanghaoyu6661/Hand-Drawn-Circuit-Diagram-Recognition.git
cd Hand-Drawn-Circuit-Diagram-Recognition
```

If you already cloned without submodules, initialize them afterward:

```bash
git submodule update --init --recursive
```

If you obtained the project through a GitHub zip/tar download, you should manually verify that `third_party/mmpose/` contains the expected config tree before running the pipeline.

---

## 3. Create the Conda Environment

Use one of the two provided Conda environments:

- `environment.yml`: recommended public reproduction environment
- `environment.full.yml`: stricter pinned environment for closer reproduction on Ubuntu + CUDA 11.8

Recommended setup:

```bash
conda env create -f environment.yml
conda activate hcd_pipeline_v2
```

Pinned variant:

```bash
conda env create -f environment.full.yml
conda activate hcd_pipeline_v2
```

If Conda is not initialized in your shell:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hcd_pipeline_v2
```

### Post-create OpenMMLab step

After the environment is created, install the OpenMMLab wheel stack:

```bash
pip install -U openmim
mim install "mmcv==2.1.0"
pip install "mmengine==0.10.7" "mmdet==3.3.0" "mmpose==1.3.2" "mmpretrain==1.2.0"
```

These commands are intentionally kept as a post-create step because this has proven more stable on fresh-machine reproduction than forcing the entire OpenMMLab stack through a single Conda environment solve.

---

## 4. Portable Path Configuration

This repository uses a **portable YAML configuration scheme**.

### Default behavior

The default files:

```text
configs/paths.yaml
configs/config.yaml
```

use **repository-relative paths**, and the shared loader:

```text
src/pipeline/config_utils.py
```

automatically resolves them against the repository root.

That means:

- you do **not** need to replace `__PROJECT_ROOT__`
- you do **not** need to hard-code `/root/autodl-tmp/...`
- if the repository structure is kept unchanged, the default config can be used directly after cloning

### Optional local override

If you want machine-specific overrides, create:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

Then edit only the entries you want to override. The pipeline automatically prefers:

```text
configs/paths.local.yaml
```

over:

```text
configs/paths.yaml
```

This is useful when your local weight locations or output locations differ from the default repository layout.

---

## 5. Download Model Weights

Pretrained model weights must be prepared separately.

Released public weights are hosted at:

```text
https://huggingface.co/why0722/hcd-circuit-weights
```

Download the required files from the Hugging Face repository and place them into the local `weights/` directory according to the layout below.

Expected directory layout:

```text
weights/
├── yolo/
│   └── best.pt
│
├── parseq/
│   └── best_parseq.pt
│
├── hawp/
│   └── last_epoch_035.pth
│
├── dinov2/
│   ├── best_bjt_partial_ft.pt
│   ├── best_mosfet_partial_ft.pt
│   └── best_dcsrc_partial_ft.pt
│
└── vitpose/
    ├── best_coco_AP_epoch_29.pth
    └── best_coco_AP_epoch_30.pth
```

If you keep the default repository structure, place the weights under `weights/` as shown above. If your local layout differs, override the corresponding entries in `configs/paths.local.yaml`.

### Note on HAWP configuration

The HAWP checkpoint is stored under:

- `weights/hawp/last_epoch_035.pth`

The HAWP runtime config is stored in the repository config directory:

- `configs/config.yaml`

So the pipeline no longer expects `weights/hawp/config.yaml`. If you reproduce the project from scratch, make sure both the released HAWP checkpoint and the repository-side `configs/config.yaml` are present.

---

## 6. Prepare Input Images

Place circuit images into:

```text
data/inputs/
```

Typical supported formats include:

```text
png
jpg
jpeg
```

Each image will be processed independently.

### Included example inputs and ground truth

This repository already provides a small runnable example set for quick verification:

- `data/inputs/` contains **5 hand-drawn circuit images**:
  - `cir1.jpg`
  - `cir2.jpg`
  - `cir3.jpg`
  - `cir4.jpg`
  - `cir5.jpg`
- `data/ground_truth/` contains the **corresponding ground-truth annotations**:
  - `cir1.gt.json`
  - `cir2.gt.json`
  - `cir3.gt.json`
  - `cir4.gt.json`
  - `cir5.gt.json`

These 5 bundled examples can be used directly to:

1. run the full pipeline end to end, and
2. run the final evaluation script against the provided GT annotations.

So even without preparing your own images first, you can use the repository-provided sample set to verify that the public release is configured correctly.

---

## 7. Run the Full Pipeline

Execute the complete pipeline:

```bash
bash scripts/run_all.sh
```

The current full pipeline includes these execution stages:

1. YOLOv10 component detection
2. image scanification / crop generation
3. PARSeq text recognition and fusion
4. component-region suppression for topology reasoning
5. HAWP junction detection
6. junction merging and wire enhancement
7. connectivity inference, subtype refinement, and port semantic inference
8. final structured JSON generation and visualization export
9. SPICE netlist generation

The shell entry script automatically:

- locates the repository root
- prefers `configs/paths.local.yaml` when present
- otherwise falls back to `configs/paths.yaml`
- resolves configuration paths through the shared portable config loader

If you are running from a shell session where local third-party packages are not already discoverable, set:

```bash
export PYTHONPATH="$PWD/third_party/HAWP:$PWD/third_party/parseq-main:$PYTHONPATH"
```

before launching `run_all.sh`.

---

## 8. Main Output Files

After the pipeline completes, outputs are generated under:

```text
outputs/run1/
```

Typical outputs:

```text
outputs/run1/
│
├── final_result/
│   ├── img/
│   │   └── *_final.png
│   └── json/
│       └── *.final.json
│
└── spice_netlists/
    └── *.spice
```

Intermediate outputs from detection, OCR, HAWP, merged points, link inference, and port classification are also stored under the same run directory.

---

## 9. SPICE Netlist Generation

The final exporter:

```text
src/pipeline/build_spice_netlists.py
```

reads:

```text
outputs/run1/final_result/json/*.final.json
```

and generates:

```text
outputs/run1/spice_netlists/*.spice
```

Supported native SPICE components include:

- resistors
- capacitors
- inductors
- voltage sources
- current sources
- diodes
- BJTs
- MOSFETs

For abstract or high-level components, placeholder `.SUBCKT` definitions are emitted when needed so the resulting netlist remains syntactically complete.

---

## 10. Notes

Important implementation details:

- OCR-recognized values such as `10k`, `100mH`, and `9V` are propagated into generated netlists when possible.
- Ground-like symbols such as `gnd` or `vss` are normalized to SPICE node `0`.
- All pipeline modules share the same YAML-driven configuration system.
- The portable configuration design makes the project easier to clone and run on a different machine without rewriting hard-coded absolute paths.

---

## 11. Evaluation

Evaluation-related scripts and reports are organized under:

```text
scripts/eval/
reports/eval/
data/ground_truth/
```

The repository already includes a small paired example set for evaluation:

- input images: `data/inputs/cir1.jpg` to `data/inputs/cir5.jpg`
- GT annotations: `data/ground_truth/cir1.gt.json` to `data/ground_truth/cir5.gt.json`

After running the pipeline on these 5 example images, you can use the evaluation script in `scripts/eval/evaluate_dimensions.py` to compare the generated final JSON outputs against the provided GT files.

This makes the repository self-contained for a quick end-to-end smoke test of both inference and evaluation.

---

## 12. Citation

If you use this project in research, please cite the corresponding paper and repository release.

---

## 13. License

This project is released for research purposes. Please refer to the repository for license details and any third-party license requirements.
