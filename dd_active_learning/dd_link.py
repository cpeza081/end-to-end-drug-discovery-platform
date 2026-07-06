#!/usr/bin/env python3
"""
dd_link.py
==========
Bridges dd_prep (library preparation) and dd_active_learning (the DD
iterative loop).  This is the connective layer. dd_prep produces a work_dir; dd_active_learning's 
campaign.yaml needs library.smiles_dir and library.fingerprint_dir pointed at the right place
inside it.  This script does that wiring automatically.

dd_prep's output layout:
    <work_dir>/library_prepared/      <- library.smiles_dir
    <work_dir>/library_prepared_fp/   <- library.fingerprint_dir
    <work_dir>/.checkpoint.json       <- used here to confirm prep finished
    <work_dir>/run_config.yaml        <- the exact config dd_prep used

Three ways to use this:

  1. Link an ALREADY-FINISHED dd_prep run to a campaign config:

       python dd_link.py --prep-work-dir /scratch/me/dd_prep_output \\
                          --campaign campaign.yaml

  2. Run dd_prep from scratch, then link automatically once it finishes:

       python dd_link.py --run-prep --prep-config my_run.yaml \\
                          --campaign campaign.yaml

  3. Just check whether a work_dir is ready to be linked (no changes made):

       python dd_link.py --prep-work-dir /scratch/me/dd_prep_output \\
                          --campaign campaign.yaml --check-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dd_utils import load_config, count_molecules_in_dir

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# dd_prep output inspection
# =============================================================================

class PrepRunStatus:
    """
    Reads dd_prep's on-disk state and reports whether it's ready to be linked to a dd_active_learning campaign.
    """

    REQUIRED_STEPS = ("organize", "fingerprint")

    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.smiles_dir = self.work_dir / "library_prepared"
        self.fp_dir = self.work_dir / "library_prepared_fp"
        self.checkpoint_path = self.work_dir / ".checkpoint.json"

    def checkpoint(self) -> dict:
        # This reads the saved progress file so the script can tell what has already finished.
        if not self.checkpoint_path.is_file():
            return {}
        with open(self.checkpoint_path) as f:
            return json.load(f)

    def is_ready(self) -> tuple[bool, list[str]]:
        """
        Returns (ready, problems). dd_active_learning needs library_prepared/
        and library_prepared_fp/ to both be populated, which correspond to
        dd_prep's "organize" and "fingerprint" steps respectively.
        """
        problems = []
        ckpt = self.checkpoint()

        for step in self.REQUIRED_STEPS:
            if ckpt.get(step) != "done":
                problems.append(
                    f"dd_prep step '{step}' has not completed "
                    f"(checkpoint shows: {ckpt.get(step, 'not started')})"
                )

        if not self.smiles_dir.is_dir() or not any(self.smiles_dir.glob("*.txt")):
            problems.append(f"No prepared SMILES files found in {self.smiles_dir}")

        if not self.fp_dir.is_dir() or not any(self.fp_dir.glob("*.txt")):
            problems.append(f"No fingerprint files found in {self.fp_dir}")

        return (len(problems) == 0, problems)

    def molecule_count(self) -> int | None:
        """Count molecules across all prepared SMILES chunks for a sanity-check.

        Uses the cached, buffered counter in dd_utils.
        """
        return count_molecules_in_dir(self.smiles_dir)


# =============================================================================
# Running dd_prep (optional path)
# =============================================================================

def run_dd_prep(prep_config: str, dry_run: bool = False) -> int:
    """
    Invoke dd_prep's CLI entry point (the installed `dd-prep` console
    script; see pyproject.toml's [project.scripts] section). We shell out
    to it, so this script works the same way whether dd_prep is installed 
    in this environment or a different one.
    """
    cmd = ["dd-prep", "--config", prep_config]
    if dry_run:
        cmd.append("--dry-run")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def get_prep_work_dir(prep_config: str) -> str:
    """Read work_dir out of a dd_prep config file."""
    with open(prep_config) as f:
        cfg = yaml.safe_load(f) or {}
    work_dir = cfg.get("work_dir")
    if not work_dir:
        raise ValueError(f"No work_dir found in {prep_config}")
    return work_dir


# =============================================================================
# Linking: write the resolved paths into the campaign config
# =============================================================================

def link_campaign(campaign_path: str, smiles_dir: str, fp_dir: str) -> None:
    """
    Update library.smiles_dir and library.fingerprint_dir in a campaign.yaml,
    preserving comments and formatting for every other line.

    A targeted text substitution is used instead of a full YAML
    load-modify-dump round trip. PyYAML's dumper does not preserve comments,
    and campaign.yaml is meant to be a hand-edited, documented file,
    so round-tripping it through yaml.dump() would strip every
    comment in it.
    """
    path = Path(campaign_path)
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    in_library_block = False
    smiles_set = fp_set = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "library:":
            in_library_block = True
            continue

        if in_library_block:
            # A library: block ends at the next top-level (non-indented) key.
            if line and not line[0].isspace() and stripped:
                in_library_block = False
                continue

            if stripped.startswith("smiles_dir:"):
                comment = _trailing_comment(line)
                lines[i] = f'  smiles_dir: "{smiles_dir}"{comment}\n'
                smiles_set = True
            elif stripped.startswith("fingerprint_dir:"):
                comment = _trailing_comment(line)
                lines[i] = f'  fingerprint_dir: "{fp_dir}"{comment}\n'
                fp_set = True

    if not (smiles_set and fp_set):
        raise ValueError(
            "Could not find library.smiles_dir / library.fingerprint_dir "
            f"in {campaign_path}. Has the file's structure changed?"
        )

    path.write_text("".join(lines))


def _trailing_comment(line: str) -> str:
    """Preserve any '  # comment' suffix already on a config line."""
    if "#" in line:
        return "  " + line[line.index("#"):].rstrip()
    return ""


# =============================================================================
# CLI
# =============================================================================

def main():
    # This is the command line entry point for linking or checking a preparation run.
    parser = argparse.ArgumentParser(
        description="Link a dd_prep run to a dd_active_learning campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--campaign", required=True,
                        help="Path to the dd_active_learning campaign.yaml to update")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prep-work-dir",
                        help="Path to an existing (already-finished) dd_prep work_dir")
    source.add_argument("--prep-config",
                        help="Path to a dd_prep config.yaml - work_dir is read from it")

    parser.add_argument("--run-prep", action="store_true",
                        help="Run dd_prep first (requires --prep-config), "
                             "then link automatically once it finishes")
    parser.add_argument("--check-only", action="store_true",
                        help="Only report readiness; do not modify campaign.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pass --dry-run through to dd_prep if --run-prep is used")
    args = parser.parse_args()

    if args.run_prep and not args.prep_config:
        parser.error("--run-prep requires --prep-config")

    # ── Resolve work_dir ──────────────────────────────────────────────────
    if args.run_prep:
        print(f"\n{'='*60}")
        print("  Running dd_prep")
        print(f"{'='*60}\n")
        rc = run_dd_prep(args.prep_config, dry_run=args.dry_run)
        if rc != 0:
            print(f"\ndd_prep exited with code {rc} - not linking.", file=sys.stderr)
            sys.exit(rc)
        work_dir = get_prep_work_dir(args.prep_config)

    elif args.prep_config:
        work_dir = get_prep_work_dir(args.prep_config)

    else:
        work_dir = args.prep_work_dir

    # ── Check readiness ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Checking dd_prep output: {work_dir}")
    print(f"{'='*60}\n")

    status = PrepRunStatus(work_dir)
    ready, problems = status.is_ready()

    if not ready:
        print("  NOT READY to link. Problems found:")
        for p in problems:
            print(f"    [FAIL] {p}")
        print()
        if args.run_prep:
            print("  dd_prep reported success but its output does not look "
                  "complete. Check dd_prep's own log for errors.")
        sys.exit(1)

    n_mols = status.molecule_count()
    print(f"  [OK] library_prepared:    {status.smiles_dir}")
    print(f"  [OK] library_prepared_fp: {status.fp_dir}")
    if n_mols is not None:
        print(f"  [OK] Total molecules:     {n_mols:,}")

    if args.check_only:
        print("\n  --check-only set: campaign.yaml was not modified.")
        return

    # ── Link ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Updating {args.campaign}")
    print(f"{'='*60}\n")

    link_campaign(
        args.campaign,
        smiles_dir=str(status.smiles_dir.resolve()),
        fp_dir=str(status.fp_dir.resolve()),
    )

    print(f"  [OK] library.smiles_dir      -> {status.smiles_dir.resolve()}")
    print(f"  [OK] library.fingerprint_dir -> {status.fp_dir.resolve()}")
    print(f"\n  Next step: python dd_validate.py --config {args.campaign}\n")


if __name__ == "__main__":
    main()