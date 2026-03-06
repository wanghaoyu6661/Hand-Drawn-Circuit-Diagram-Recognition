# Hand-Drawn-Circuit-Diagram-Recognition

Deep learning–based recognition and structural parsing pipeline for **hand‑drawn circuit diagrams**.

This project provides an **end‑to‑end framework** that converts hand‑drawn circuit schematics into structured machine‑readable representations, including:

- intermediate outputs for each processing stage
- final structured JSON (`*.final.json`)
- visualization images (`*_final.png`)
- SPICE netlists (`*.spice`) ready for circuit simulation tools

The system combines deep learning perception modules with structural reasoning to reconstruct circuit topology and semantics from noisy hand‑drawn inputs.

---

# 1. Repository Structure

Typical repository layout:

```
Hand-Drawn-Circuit-Diagram-Recognition/
│
├── assets/                 # static resources (fonts, meta files)
├── configs/                # YAML configuration files
│   ├── paths.yaml
│   └── paths.example.yaml
│
├── data/
│   └── inputs/             # input circuit images
│
├── scripts/
│   └── run_all.sh          # full pipeline entry point
│
├── src/
│   └── pipeline/           # main processing modules
│
├── third_party/            # external repositories (HAWP, PARSeq, MMPose)
│
├── environment.yml         # conda environment definition
└── README.md
```

---

# 2. Clone the Repository

Clone the public repository:

```bash
git clone https://github.com/wanghaoyu6661/Hand-Drawn-Circuit-Diagram-Recognition.git
cd Hand-Drawn-Circuit-Diagram-Recognition
```

Initialize third‑party submodules:

```bash
git submodule update --init --recursive
```

---

# 3. Create the Conda Environment

Create the project environment using the provided YAML file:

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

# 4. Configure Paths

This project uses a YAML‑based configuration system.

We recommend creating a **local configuration file** instead of modifying the default repository configuration.

### Step 1

Copy the example configuration:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

### Step 2

Edit the file and replace

```
__PROJECT_ROOT__
```

with the **absolute path of your local repository**.

Example:

```
/home/user/Hand-Drawn-Circuit-Diagram-Recognition
```

The pipeline automatically prefers `paths.local.yaml` if it exists.

---

# 5. Download Model Weights

Pretrained model weights must be downloaded separately.

Example directory layout expected by the pipeline:

```
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

Place the weights in the directories specified by `configs/paths.local.yaml`.

---

# 6. Prepare Input Images

Place circuit images into:

```
data/inputs/
```

Supported formats include:

```
png
jpg
jpeg
```

Each image will be processed independently by the pipeline.

---

# 7. Run the Full Pipeline

Execute the complete recognition pipeline:

```bash
bash scripts/run_all.sh
```

The pipeline automatically performs:

1. YOLOv10 component detection
2. PARSeq text recognition
3. component suppression for topology reasoning
4. HAWP junction detection
5. junction merging
6. wire enhancement
7. connectivity inference
8. component subtype classification
9. port semantic inference
10. final structured JSON generation
11. SPICE netlist export

---

# 8. Main Output Files

After the pipeline completes, outputs are generated under:

```
outputs/run1/
```

Example:

```
outputs/run1/
│
├── final_result/
│   ├── img/
│   │   └── *_final.png
│   │
│   └── json/
│       └── *.final.json
│
└── spice_netlists/
    └── *.spice
```

---

# 9. SPICE Netlist Generation

The final step automatically converts structured circuit JSON files into SPICE netlists.

The exporter:

```
src/pipeline/build_spice_netlists.py
```

reads

```
outputs/run1/final_result/json/*.final.json
```

and generates

```
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

For abstract or high‑level components (logic gates, IC blocks, etc.), placeholder `.SUBCKT` definitions are automatically generated so the resulting netlist remains syntactically complete.

---

# 10. Notes

Important implementation details:

- OCR‑recognized text values such as `10k`, `100mH`, or `9V` are propagated into the generated netlists when possible.
- Ground symbols (`gnd`, `vss`) are normalized to SPICE node `0`.
- The pipeline uses a unified YAML configuration system shared across all modules.

---

# 11. Citation

If you use this project in research, please cite the corresponding paper.

---

# 12. License

This project is released for research purposes. Please check the repository for license information.
