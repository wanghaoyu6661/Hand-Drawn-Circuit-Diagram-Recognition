from pathlib import Path
import yaml
import os

def _project_root() -> Path:
    # src/pipeline/path_config.py -> project root
    return Path(__file__).resolve().parents[2]

def load_paths():
    cfg_path = _project_root() / "configs" / "paths.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_paths()

def cfg_get(*keys, default=None):
    cur = CFG
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

PROJECT_ROOT = str(_project_root())