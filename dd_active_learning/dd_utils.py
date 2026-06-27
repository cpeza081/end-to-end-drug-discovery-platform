"""
dd_utils.py
===========
Shared utilities for the DD orchestration scripts.

Kept intentionally small such that only logic that is used by more than
one script lives here.  Everything domain-specific stays in its own file.
"""

import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """
    Load a campaign YAML file and expand environment variables ($VAR / ${VAR})
    in every string value, recursively.

    Returns the fully resolved config as a plain dict.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand_env(raw)


def _expand_env(obj):
    """Recursively walk a parsed YAML structure and expand env vars in strings."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj