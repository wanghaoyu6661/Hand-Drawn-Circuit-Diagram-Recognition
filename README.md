# Hand-Drawn-Circuit-Diagram-Recognition

Deep learning-based recognition and structural parsing pipeline for hand-drawn circuit diagrams.

> **Current status**
> - ✅ Core pipeline reconstructed and runnable locally (YAML-driven).
> - 🟡 Public reproducibility release in progress (environment export / weight download instructions / clean-room validation).

---

## 1. Project Overview

This project provides an end-to-end pipeline for recognizing hand-drawn circuit diagrams and generating structured outputs (including intermediate results and final JSON visualization).

The main pipeline is executed by:

```bash
bash scripts/run_all.sh
Quick Start
Clone the repository:
git clone git@github.com:wanghaoyu6661/Hand-Drawn-Circuit-Diagram-Recognition.git
cd Hand-Drawn-Circuit-Diagram-Recognition
Initialize the mmpose submodule (recommended shallow clone):
git submodule update --init --depth 1
If you prefer full recursive clone:
git submodule update --init --recursive
