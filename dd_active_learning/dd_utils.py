"""
dd_utils.py
===========
Shared utilities for the DD orchestration scripts.

Kept intentionally small such that only logic that is used by more than
one script lives here.  Everything domain-specific stays in its own file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

# Shared helpers for the active learning scripts.


def count_lines_fast(path) -> int:
    """
    Count newlines in a file using buffered binary reads..
    """
    count = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB at a time - constant memory
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def count_molecules_in_dir(directory, pattern: str = "*.txt",
                           use_cache: bool = True) -> int | None:
    """
    Total line count across files matching ``pattern`` in ``directory``.

    Results are memoised in a hidden sidecar file (``.mol_count_cache.json``)
    keyed on each file's (size, mtime).  A directory whose files have not
    changed is counted once and reused on every subsequent call.

    Returns None if the directory does not exist, otherwise the total
    (0 for an existing directory with no matching files).
    """
    d = Path(directory)
    if not d.is_dir():
        return None

    files = sorted(f for f in d.glob(pattern) if f.is_file())
    if not files:
        return 0

    cache_path = d / ".mol_count_cache.json"
    cached_entries: dict = {}
    if use_cache and cache_path.is_file():
        try:
            cached_entries = json.loads(cache_path.read_text()).get("files", {})
        except (json.JSONDecodeError, OSError):
            cached_entries = {}

    new_entries: dict = {}
    total = 0
    changed = False
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        sig = f"{st.st_size}:{st.st_mtime_ns}"
        prev = cached_entries.get(f.name)
        if prev and prev.get("sig") == sig:
            lines = prev["lines"]
        else:
            try:
                lines = count_lines_fast(f)
            except OSError:
                continue
            changed = True
        new_entries[f.name] = {"sig": sig, "lines": lines}
        total += lines

    if use_cache and (changed or new_entries.keys() != cached_entries.keys()):
        try:
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"files": new_entries}))
            os.replace(tmp, cache_path)
        except OSError:
            pass  # e.g. read-only directory - counting still succeeds

    return total


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