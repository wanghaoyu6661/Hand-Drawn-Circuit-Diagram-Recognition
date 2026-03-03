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