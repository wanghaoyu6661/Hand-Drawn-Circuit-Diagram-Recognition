from __future__ import annotations

from config_utils import get_default_config_path, get_repo_root, load_paths_config, project_path


def _project_root() -> str:
    return str(get_repo_root())


def load_paths(cfg_path=None):
    return load_paths_config(cfg_path)


CFG = load_paths()


def reload_paths(cfg_path=None):
    global CFG
    CFG = load_paths(cfg_path)
    return CFG


def cfg_get(*keys, default=None, cfg=None):
    cur = CFG if cfg is None else cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


PROJECT_ROOT = _project_root()
DEFAULT_CFG_PATH = str(get_default_config_path())
