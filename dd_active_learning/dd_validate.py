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
from dd_orchestrator import _docking_program


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

        # Docking engine + receptor / binding-site inputs
        try:
            program = _docking_program(self.cfg)
            self.ok(f"Docking program: {program}")
        except ValueError as exc:
            self.fail(str(exc))
            program = None

        receptor = dock.get("receptor_file")
        if receptor and Path(receptor).exists():
            self.ok(f"Receptor: {receptor}")
        else:
            self.fail(f"Receptor file not found: {receptor}")

        # The binding box is produced once by dd_receptor_prep.py.
        box_json = dock.get("box_json")
        if box_json and Path(box_json).exists():
            self.ok(f"Binding box: {box_json}")
        else:
            self.fail(f"Binding box not found: {box_json} "
                      f"(run: python dd_receptor_prep.py --config <campaign.yaml>)")

        # Site strategy inputs (checked for the configured method only).
        site = dock.get("site", {})
        method = site.get("method", "reference_ligand")
        if method == "reference_ligand":
            ref = site.get("reference_ligand")
            if ref and Path(ref).exists():
                self.ok(f"Reference ligand: {ref}")
            else:
                self.fail(f"site.method=reference_ligand but reference ligand "
                          f"not found: {ref}")
        elif method == "p2rank":
            self.ok("Site strategy: p2rank (pocket predicted from receptor)")
        elif method == "manual":
            center, size = site.get("center"), site.get("size")
            ok_shape = (isinstance(center, (list, tuple)) and len(center) == 3
                        and isinstance(size, (list, tuple)) and len(size) == 3)
            if ok_shape and all(s > 0 for s in size):
                self.ok(f"Site strategy: manual (center={center}, size={size})")
            else:
                self.fail("site.method=manual requires center [x,y,z] and "
                          "size [x,y,z] (three positive numbers each)")
        else:
            self.warn(f"Unknown site.method '{method}' - dd_receptor_prep.py "
                      f"will reject it unless a matching strategy is registered")

        # AutoDock-GPU needs pre-computed grid maps.
        if program == "AUTODOCK_GPU":
            maps = dock.get("maps_fld")
            if maps and Path(maps).exists():
                self.ok(f"AutoDock-GPU maps: {maps}")
            else:
                self.fail(f"AutoDock-GPU maps (.fld) not found: {maps} "
                          f"(run dd_receptor_prep.py to build them)")

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

    # ------------------------------------------------------------------
    def check_tools(self):
        print("\n-- Tools -------------------------------------------")
        try:
            program = _docking_program(self.cfg)
        except ValueError:
            program = None

        # Docking-engine binaries (must be on PATH inside the conda env).
        dock = self.cfg["docking"]
        if program == "GNINA":
            docking_tools = ["gnina"]
        elif program == "AUTODOCK_GPU":
            # The docking binary plus the receptor-prep toolchain.
            docking_tools = [dock.get("autodock_bin", "autodock_gpu_128wi"),
                             "autogrid4", "mk_prepare_receptor.py"]
        else:
            docking_tools = []

        # P2Rank is only needed when that site strategy is selected.
        if dock.get("site", {}).get("method") == "p2rank":
            docking_tools.append(dock["site"].get("p2rank_exec", "prank"))

        # If the docking tools are provided by cluster modules, they will not
        # be on PATH here (login node), that's expected. 
        modules = self.cfg["env"].get("modules") or []
        for tool in docking_tools:
            if shutil.which(tool):
                self.ok(f"Tool on PATH: {tool}")
            elif modules:
                self.warn(f"{tool} not on PATH now (expected if it comes from a "
                          f"module: {', '.join(modules)}); jobs will module-load it")
            else:
                self.warn(f"Tool not found on PATH: {tool} "
                          f"(must be available on the compute node at run time)")

        if modules:
            self.ok(f"Cluster modules to load in jobs: {', '.join(modules)}")

        # Scheduler command
        sched = self.cfg["scheduler"]["type"].upper()
        submit_cmd = {"SLURM": "sbatch", "PBS": "qsub", "SGE": "qsub"}[sched]
        if shutil.which(submit_cmd):
            self.ok(f"Scheduler submit command: {submit_cmd}")
        else:
            self.warn(f"Scheduler command not found on PATH: {submit_cmd} "
                      f"(OK if running from a login node)")

        # import Python packages through the campaign's env exactly as a
        # job would, not in this interpreter.
        env_cfg = self.cfg["env"]
        activate = env_cfg.get("activate", "")
        modules = env_cfg.get("modules") or []
        job_pkgs = "import numpy, pandas, sklearn, rdkit, meeko, tensorflow"
        if activate:
            parts = []
            if modules:
                parts.append("module load " + " ".join(modules))
            parts.append(activate)
            parts.append(f'python -c "{job_pkgs}"')
            cmd = " && ".join(parts)
            try:
                r = subprocess.run(["bash", "-lc", cmd],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode == 0:
                    self.ok("Campaign env imports OK (numpy, pandas, sklearn, "
                            "rdkit, meeko, tensorflow)")
                else:
                    last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
                    self.fail(f"Campaign env is missing a package -> {last}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                self.warn(f"Could not test campaign-env imports ({exc}); "
                          f"activate the env and run: python -c '{job_pkgs}'")
        else:
            # No activation configured. fall back to this interpreter.
            for pkg in ("yaml", "pandas", "numpy", "rdkit", "meeko"):
                try:
                    __import__(pkg)
                    self.ok(f"Python package: {pkg}")
                except ImportError:
                    self.fail(f"Python package not importable: {pkg}")

    # ------------------------------------------------------------------
    def check_environment(self):
        print("\n-- Environment -------------------------------------")

        # Python environment: either a conda env or a virtualenv, depending on
        # how env.activate was set up by the wizard.
        env_cfg = self.cfg["env"]
        activate = env_cfg.get("activate", "") or ""
        conda_env = env_cfg.get("conda_env", "") or ""

        if "bin/activate" in activate:
            # virtualenv: find the activate script path in the command
            venv_path = next((t for t in activate.replace('"', " ").split()
                              if t.endswith("bin/activate")), None)
            if venv_path and Path(venv_path).exists():
                self.ok(f"Python virtualenv found: {venv_path}")
            elif venv_path:
                self.warn(f"virtualenv not created yet: {venv_path} "
                          f"(re-run setup_active_learning.sh to build it)")
            else:
                self.ok("Using a Python virtualenv (env.activate set)")
        elif conda_env:
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
        else:
            self.warn("No env.activate or env.conda_env set; how will jobs load "
                      "the Python environment?")

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
