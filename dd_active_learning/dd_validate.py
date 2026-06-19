#!/usr/bin/env python3
"""
dd_validate.py
==============
Pre-flight checks for a Deep Docking campaign.

Validates that all paths, tools, and environment requirements are met
BEFORE submitting any jobs to the cluster.  Run this once after editing
campaign.yaml to catch configuration mistakes early.

Usage:
  python dd_validate.py --config campaign.yaml
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dd_utils import load_config


class Validator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.errors = []
        self.warnings = []
        self.passed = []

    def ok(self, msg):
        self.passed.append(msg)
        print(f"  [OK]  {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    def fail(self, msg):
        self.errors.append(msg)
        print(f"  [FAIL]  {msg}")

    # ------------------------------------------------------------------
    def check_paths(self):
        print("\n-- Paths -------------------------------------------")
        lib = self.cfg["library"]
        dock = self.cfg["docking"]
        env = self.cfg["env"]

        # Library dirs must exist and be non-empty
        for label, path in [
            ("smiles_dir",      lib["smiles_dir"]),
            ("fingerprint_dir", lib["fingerprint_dir"]),
        ]:
            p = Path(path)
            if not p.exists():
                self.fail(f"{label} does not exist: {path}")
            elif not any(p.iterdir()):
                self.warn(f"{label} exists but is empty: {path}")
            else:
                n = sum(1 for _ in p.iterdir())
                self.ok(f"{label}: {path}  ({n} files)")

        # Grid file must exist
        grid = dock["grid_file"]
        if Path(grid).exists():
            self.ok(f"Docking grid: {grid}")
        else:
            self.fail(f"Docking grid not found: {grid}")

        # DD protocol repo
        dd = env["dd_protocol_dir"]
        required_dd_scripts = [
            "scripts_1/molecular_file_count_updated.py",
            "scripts_1/sampling.py",
            "scripts_1/sanity_check.py",
            "scripts_1/extracting_morgan.py",
            "scripts_1/extracting_smiles.py",
            "scripts_2/extract_labels.py",
            "scripts_2/simple_job_models_manual.py",
            "scripts_2/hyperparameter_result_evaluation.py",
            "scripts_2/simple_job_predictions_manual.py",
            "utilities/final_extraction.py",
        ]
        dd_ok = True
        for script in required_dd_scripts:
            full = Path(dd) / script
            if not full.exists():
                self.fail(f"DD script missing: {full}")
                dd_ok = False
        if dd_ok:
            self.ok(f"DD protocol dir: {dd}  (all scripts found)")

        # OpenEye dir
        oe = env["openeye_dir"]
        if Path(oe).exists():
            self.ok(f"OpenEye dir: {oe}")
        else:
            self.fail(f"OpenEye dir not found: {oe}")

    # ------------------------------------------------------------------
    def check_tools(self):
        print("\n-- Tools -------------------------------------------")
        oe = self.cfg["env"]["openeye_dir"]
        program = self.cfg["docking"]["program"].upper()

        # OpenEye tools
        oe_tools = ["flipper", "tautomers", "oeomega"]
        if program == "FRED":
            oe_tools.append("fred")

        for tool in oe_tools:
            full = Path(oe) / tool
            if full.exists():
                self.ok(f"OpenEye tool found: {tool}")
            else:
                # Also check $PATH
                if shutil.which(tool):
                    self.ok(f"OpenEye tool on PATH: {tool}")
                else:
                    self.fail(f"OpenEye tool not found: {tool}")

        # Scheduler command
        sched = self.cfg["scheduler"]["type"].upper()
        submit_cmd = {"SLURM": "sbatch", "PBS": "qsub", "SGE": "qsub"}[sched]
        if shutil.which(submit_cmd):
            self.ok(f"Scheduler submit command: {submit_cmd}")
        else:
            self.warn(f"Scheduler command not found on PATH: {submit_cmd} "
                      f"(OK if running from a login node)")

        # Python packages
        packages = ["yaml", "pandas", "rdkit"]
        for pkg in packages:
            try:
                __import__(pkg)
                self.ok(f"Python package: {pkg}")
            except ImportError:
                self.fail(f"Python package not importable: {pkg}")

    # ------------------------------------------------------------------
    def check_environment(self):
        print("\n-- Environment -------------------------------------")

        # OE_LICENSE
        oe_lic = os.environ.get("OE_LICENSE", "")
        oe_dir = self.cfg["env"]["openeye_dir"]
        local_lic = Path(oe_dir) / "oe_license.txt"

        if oe_lic and Path(oe_lic).exists():
            self.ok(f"OE_LICENSE (env var): {oe_lic}")
        elif local_lic.exists():
            self.ok(f"OE_LICENSE (local): {local_lic}")
        else:
            self.fail("OpenEye license not found (OE_LICENSE env var or "
                      "<openeye_dir>/oe_license.txt)")

        # Conda env
        conda_env = self.cfg["env"]["conda_env"]
        try:
            result = subprocess.run(
                ["conda", "env", "list"], capture_output=True, text=True, timeout=15
            )
            if conda_env in result.stdout:
                self.ok(f"Conda environment found: {conda_env}")
            else:
                self.fail(f"Conda environment not found: {conda_env}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.warn("Cannot verify conda environment (conda not on PATH or timeout)")

        # SCRATCH / project dir parent writeable
        proj = Path(self.cfg["project_dir"])
        parent = proj.parent
        if parent.exists() and os.access(parent, os.W_OK):
            self.ok(f"Project parent dir is writable: {parent}")
        elif not parent.exists():
            self.warn(f"Project parent dir does not exist yet: {parent} "
                      f"(will be created on launch)")
        else:
            self.fail(f"Project parent dir is not writable: {parent}")

    # ------------------------------------------------------------------
    def check_dd_params(self):
        print("\n-- DD parameters -----------------------------------")
        dd = self.cfg["dd"]
        total_sample = dd["train_size"] + 2 * dd["val_size"]

        if dd["val_size"] < 250_000:
            self.warn(f"val_size={dd['val_size']:,} is below the recommended "
                      f"minimum of 250,000 (see paper Section 'Molecular sample size')")
        else:
            self.ok(f"val_size={dd['val_size']:,}  (>=250K minimum)")

        if dd["num_models"] not in (16, 24, 48, 72, 144):
            self.fail(f"num_models={dd['num_models']} is not one of the "
                      f"allowed values: 16, 24, 48, 72, 144")
        else:
            self.ok(f"num_models={dd['num_models']}")

        if dd["recall"] < 0.75 or dd["recall"] > 0.95:
            self.warn(f"recall={dd['recall']} is outside the recommended "
                      f"range 0.75-0.90")
        else:
            self.ok(f"recall={dd['recall']}  (0.75-0.90 range)")

        if dd["total_iterations"] < 4:
            self.warn(f"total_iterations={dd['total_iterations']} - fewer "
                      f"than 4 iterations may give poor library reduction")
        else:
            self.ok(f"total_iterations={dd['total_iterations']}")

        print(f"\n    First iteration total sampling: {total_sample:,} molecules")
        print(f"    (train={dd['train_size']:,} + val={dd['val_size']:,} "
              f"+ test={dd['val_size']:,})")

    # ------------------------------------------------------------------
    def report(self):
        print(f"\n{'='*55}")
        print(f"  Validation summary")
        print(f"{'='*55}")
        print(f"  Passed:   {len(self.passed)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.errors)}")

        if self.errors:
            print(f"\n  Errors must be fixed before launching:")
            for e in self.errors:
                print(f"    [FAIL]  {e}")
            print()
            return False

        if self.warnings:
            print(f"\n  Warnings (review before proceeding):")
            for w in self.warnings:
                print(f"    [WARN]  {w}")

        print(f"\n  {'All checks passed.' if not self.errors else ''}")
        print(f"  Ready to launch:  python dd_orchestrator.py --config <config.yaml>\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate a Deep Docking campaign configuration"
    )
    parser.add_argument("--config", required=True,
                        help="Path to campaign YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"\nValidating campaign: {cfg['campaign_name']}")

    v = Validator(cfg)
    v.check_paths()
    v.check_tools()
    v.check_environment()
    v.check_dd_params()
    ok = v.report()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
