#!/usr/bin/env python3
"""
dd_status.py
============
Campaign progress dashboard for a running Deep Docking campaign.

Reads the campaign_state.json produced by dd_orchestrator.py and optionally
queries the scheduler to show live job status.

Usage:
  python dd_status.py --config campaign.yaml
  python dd_status.py --config campaign.yaml --no-scheduler   # offline mode
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from dd_utils import load_config, count_molecules_in_dir
from dd_orchestrator import PHASES as PHASE_NAMES  # single definition of phase labels


STATUS_SYMBOLS = {
    "RUNNING":       "[RUN] ",
    "PENDING":       "[WAIT]",
    "COMPLETED":     "[OK]  ",
    "FAILED":        "[FAIL]",
    "CANCELLED":     "[CANC]",
    "UNKNOWN":       "[??]  ",
    "TIMEOUT":       "[TIME]",
    "OUT_OF_MEMORY": "[OOM] ",
    "NODE_FAIL":     "[NODE]",
    "PREEMPTED":     "[PRE] ",
    "REQUEUED":      "[REQ] ",
    "SUSPENDED":     "[SUSP]",
    "BOOT_FAIL":     "[BOOT]",
    "DEADLINE":      "[DDL] ",
    "REVOKED":       "[REV] ",
}


# Aggregation priority: a single "bad" or in-progress task should dominate the
# summary.  Used to collapse a job array's many task states into one label.
_STATE_PRIORITY = [
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
    "CANCELLED", "REVOKED", "PREEMPTED", "SUSPENDED",
    "RUNNING", "REQUEUED", "PENDING", "COMPLETED",
]


def _aggregate_states(states: list[str]) -> str:
    """Collapse many sacct State rows into one summary label by priority."""
    norm = [s.split("+")[0].strip().upper() for s in states if s.strip()]
    if not norm:
        return "UNKNOWN"
    for st in _STATE_PRIORITY:
        if st in norm:
            return st
    return norm[0]


def query_job_status(job_id: str, sched_type: str) -> str:
    """Ask the scheduler for the current state of a job ID.

    For SLURM job arrays this aggregates across all task states.
    """
    try:
        if sched_type == "SLURM":
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                capture_output=True, text=True, timeout=15
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if lines:
                return _aggregate_states(lines)

        elif sched_type == "PBS":
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "job_state" in line and "=" in line:
                    state = line.split("=", 1)[1].strip()
                    # R=Running, Q=Queued, C=Complete, E=Exiting
                    mapping = {"R": "RUNNING", "Q": "PENDING",
                               "C": "COMPLETED", "E": "RUNNING"}
                    return mapping.get(state, "UNKNOWN")

        elif sched_type == "SGE":
            result = subprocess.run(
                ["qstat", "-j", job_id],
                capture_output=True, text=True, timeout=10
            )
            if "Following jobs do not exist" in result.stderr:
                return "COMPLETED"  # SGE removes completed jobs from qstat
            for line in result.stdout.splitlines():
                if "job_state" in line and ":" in line:
                    state = line.split(":", 1)[1].strip()
                    if state:
                        return state.upper()

    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass

    return "UNKNOWN"


def count_molecules(path: str) -> int | None:
    """Count total molecules in a directory of prediction/fingerprint files.

    Delegates to the cached counter in dd_utils.
    """
    total = count_molecules_in_dir(path)
    if total is None:
        return None
    return total if total > 0 else None


def read_best_model_stats(dd_root: str, iteration: int) -> dict:
    """Parse the best_model_stats.txt file for an iteration.

    dd_root is the DD project working directory ({project_dir}/{campaign_name}),
    and iterations are unpadded (iteration_1, not iteration_01) to match
    DD_protocol's layout.
    """
    stats_path = (Path(dd_root) / f"iteration_{iteration}"
                  / "best_model_stats.txt")
    result = {}
    if not stats_path.exists():
        return result
    try:
        text = stats_path.read_text()
        for key, pattern in [
            ("precision",       r"Model Precision:\s*([\d.]+)"),
            ("recall",          r"Model Recall:\s*([\d.]+)"),
            ("auc",             r"Model Auc:\s*([\d.]+)"),
            ("total_left",      r"Total Left(?:\s+Testing)?:\s*([\d.]+)"),
        ]:
            m = re.search(pattern, text)
            if m:
                result[key] = float(m.group(1))
    except OSError:
        pass
    return result


def render_dashboard(cfg: dict, state: dict, query_scheduler: bool):
    proj = cfg["project_dir"]
    total_iter = cfg["dd"]["total_iterations"]
    sched_type = cfg["scheduler"]["type"]
    campaign = cfg["campaign_name"]
    # DD_protocol writes under {project_dir}/{campaign_name}/iteration_{n}
    # (unpadded). 
    dd_root = f"{proj}/{campaign}"
    iters = state.get("iterations", {})

    print(f"\n{'='*65}")
    print(f"  Deep Docking Campaign: {campaign}")
    print(f"  Project: {proj}")
    print(f"  Scheduler: {sched_type}  |  Iterations: {total_iter}")
    print(f"{'='*65}")

    for it in range(1, total_iter + 1):
        it_data = iters.get(str(it), {})
        if not it_data:
            print(f"\n  Iteration {it:2d}   [not yet submitted]")
            continue

        # Check how many molecules survived inference (if done)
        pred_dir = (Path(dd_root) / f"iteration_{it}"
                    / "morgan_1024_predictions")
        n_remaining = count_molecules(str(pred_dir))

        # Model performance stats
        stats = read_best_model_stats(dd_root, it)

        print(f"\n  Iteration {it:2d}")
        print(f"  {'-'*55}")

        # Phases 1-3 are single jobs. Phase 4 is three steps (4a labels+gen,
        # 4b training array, 4c evaluate). Phase 5 is two steps (5a gen, 5b array).
        phase_rows = [(str(p), PHASE_NAMES[p], f"phase{p}_job_id")
                      for p in (1, 2, 3)]
        phase_rows.append(("4a", "Labels+gen",  "phase4a_job_id"))
        phase_rows.append(("4b", "Train array", "phase4b_job_id"))
        phase_rows.append(("4c", "Best model",  "phase4c_job_id"))
        phase_rows.append(("5a", "Pred-gen",    "phase5a_job_id"))
        phase_rows.append(("5b", "Inference",   "phase5b_job_id"))

        for label, phase_name, key in phase_rows:
            job_id = it_data.get(key)
            submitted = it_data.get(key.replace("_job_id", "_submitted"), "")

            if not job_id:
                print(f"    Phase {label:<3} ({phase_name:<12})  "
                      f"[not submitted]")
                continue

            if query_scheduler:
                raw_status = query_job_status(job_id, sched_type)
            else:
                raw_status = "UNKNOWN"

            symbol = STATUS_SYMBOLS.get(raw_status, "[??]  ")
            ts = submitted[:16] if submitted else ""
            print(f"    Phase {label:<3} ({phase_name:<12})  "
                  f"{symbol} {raw_status:<10}  job={job_id:<14}  {ts}")

        if stats:
            print(f"\n    Model performance:")
            if "auc" in stats:
                print(f"      AUC:       {stats['auc']:.4f}")
            if "precision" in stats:
                print(f"      Precision: {stats['precision']:.4f}")
            if "recall" in stats:
                print(f"      Recall:    {stats['recall']:.4f}")
            if "total_left" in stats:
                n = stats["total_left"]
                print(f"      Est. remaining: {n:,.0f}  "
                      f"({n/1e6:.1f}M)")

        if n_remaining is not None:
            print(f"    Actual prediction files: {n_remaining:,}")

    # Final extraction
    final_id = state.get("final_extraction_job_id")
    print(f"\n{'-'*65}")
    if final_id:
        if query_scheduler:
            status = query_job_status(final_id, sched_type)
        else:
            status = "UNKNOWN"
        symbol = STATUS_SYMBOLS.get(status, "[??]  ")
        print(f"  Final extraction:  {symbol} {status:<10}  job={final_id}")
    else:
        print(f"  Final extraction:  [not yet submitted]")

    # Check for output files
    for fname in ("smiles.csv", "id_score.csv"):
        p = Path(proj) / fname
        if p.exists():
            size = p.stat().st_size
            print(f"  Output ready: {fname}  ({size/1e6:.1f} MB)")

    print(f"\n{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deep Docking campaign status dashboard"
    )
    parser.add_argument("--config", required=True,
                        help="Path to campaign YAML config file")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="Skip live job status queries (offline mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proj = cfg["project_dir"]
    state_path = Path(proj) / "campaign_state.json"

    if not state_path.exists():
        print(f"No campaign state found at {state_path}")
        print("Has the campaign been launched with dd_orchestrator.py?")
        return

    with open(state_path) as f:
        state = json.load(f)

    render_dashboard(cfg, state, query_scheduler=not args.no_scheduler)


if __name__ == "__main__":
    main()