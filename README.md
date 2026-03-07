# Hand-Drawn-Circuit-Diagram-Recognition

Deep learning–based recognition and structural parsing pipeline for **hand-drawn circuit diagrams**.

This project provides an **end-to-end framework** that converts hand-drawn circuit schematics into structured machine-readable representations, including:

- intermediate outputs for each processing stage
- final structured JSON (`*.final.json`)
- visualization images (`*_final.png`)
- SPICE netlists (`*.spice`) ready for circuit simulation tools

The system combines deep learning perception modules with structural reasoning to recover circuit topology, component semantics, and simulation-oriented structure from noisy hand-drawn inputs.

---

## 1. Repository Structure

Typical repository layout:

```text
Hand-Drawn-Circuit-Diagram-Recognition/
│
├── assets/                     # static resources (fonts, metadata)
├── configs/                    # portable YAML configuration files
│   ├── paths.yaml
│   ├── paths.example.yaml
│   └── mmpose_ports/
│
├── data/
│   ├── inputs/                 # input circuit images
│   └── ground_truth/           # evaluation data (if used)
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
├── environment.yml             # conda environment definition
└── README.md
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/wanghaoyu6661/Hand-Drawn-Circuit-Diagram-Recognition.git
cd Hand-Drawn-Circuit-Diagram-Recognition
```

If your clone does not yet contain the third-party dependencies, initialize them as needed:

```bash
git submodule update --init --recursive
```

---

## 3. Create the Conda Environment

Create the project environment:

```bash
conda env create -f environment.yml
conda activate hcd_pipeline_v2
```

If Conda is not initialized in your shell:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hcd_pipeline_v2
```

---

## 4. Portable Path Configuration

This repository now uses a **portable YAML configuration scheme**.

### Default behavior

The default file:

```text
configs/paths.yaml
```

uses **repository-relative paths**, and the shared loader:

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
│   ├── config.yaml
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

You can adapt these components to benchmark final JSON outputs against ground-truth annotations.

---

## 12. Citation

If you use this project in research, please cite the corresponding paper and repository release.

---

## 13. License

This project is released for research purposes. Please refer to the repository for license details and any third-party license requirements.
