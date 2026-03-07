from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PATH_SECTIONS = ("legacy", "paths", "weights")
DEFAULT_HAWP_CONFIG_REL = Path("configs") / "config.yaml"


def get_repo_root() -> Path:
    """Return repository root based on this file location."""
    # src/pipeline/config_utils.py -> repo root
    return Path(__file__).resolve().parents[2]


def get_default_hawp_config_path(repo_root: str | Path | None = None) -> Path:
    """Return the repository-tracked default HAWP config path."""
    repo_root = Path(repo_root) if repo_root is not None else get_repo_root()
    return (repo_root / DEFAULT_HAWP_CONFIG_REL).resolve()


def get_default_config_path(repo_root: str | Path | None = None) -> Path:
    repo_root = Path(repo_root) if repo_root is not None else get_repo_root()

    env_path = os.environ.get("HCD_PATHS_CONFIG")
    if env_path:
        env_path = os.path.expanduser(env_path)
        p = Path(env_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    local_path = repo_root / "configs" / "paths.local.yaml"
    if local_path.is_file():
        return local_path

    return repo_root / "configs" / "paths.yaml"


def project_path(*parts: str) -> str:
    return str((get_repo_root().joinpath(*parts)).resolve())


def _resolve_path_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_path_value(v, repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_path_value(v, repo_root) for v in value]
    if isinstance(value, str):
        s = os.path.expanduser(value.strip())
        if not s:
            return value
        p = Path(s)
        if p.is_absolute():
            return str(p)
        return str((repo_root / p).resolve())
    return value


def load_paths_config(cfg_path: str | Path | None = None) -> dict:
    repo_root = get_repo_root()

    if cfg_path is None:
        cfg_path = get_default_config_path(repo_root)
    else:
        cfg_path = Path(os.path.expanduser(str(cfg_path)))
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML root must be a mapping: {cfg_path}")

    resolved = dict(cfg)
    resolved["project_root"] = str(repo_root)

    for section in PATH_SECTIONS:
        value = cfg.get(section, {})
        if isinstance(value, dict):
            resolved[section] = _resolve_path_value(value, repo_root)

    weights = resolved.setdefault("weights", {})
    if isinstance(weights, dict):
        weights.setdefault("hawp_cfg", str(get_default_hawp_config_path(repo_root)))

    return resolved
