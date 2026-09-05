#!/usr/bin/env python3
"""
dd_orchestrator.py
==================
Deep Docking active-learning campaign orchestrator.

Reads a YAML config file and drives the DD loop:
    Iteration 1:  Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5
    Iteration N:  Phase 1 (from previous predictions) -> Phase 2 -> 3 -> 4 -> 5
  Final:        extract SMILES of surviving virtual hits

The orchestrator submits one job per phase, using the scheduler's native
dependency mechanism.  All job IDs are logged to <project_dir>/campaign_state.json
so a crashed run can be resumed from the last completed phase.

Scheduler support
-----------------
SLURM  - full support (afterok dependencies, sbatch)
PBS    - basic support (afterok dependencies, qsub)
SGE    - basic support (hold_jid dependencies, qsub)

Usage
-----
  # Start a new campaign (or resume a crashed one):
  python dd_orchestrator.py --config campaign.yaml

  # Start from a specific iteration / phase (useful after manual fixes):
  python dd_orchestrator.py --config campaign.yaml --start-iter 3 --start-phase 4

  # Dry run: print the job scripts without submitting:
  python dd_orchestrator.py --config campaign.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from dd_utils import load_config


# =============================================================================
# Campaign state  (persisted to JSON so we can resume)
# =============================================================================

class CampaignState:
    """
    Tracks which phases have been submitted and stores their job IDs.
    Written to <project_dir>/campaign_state.json after every submission.
    """

    def __init__(self, project_dir: str):
        self.path = Path(project_dir) / "campaign_state.json"
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                # A truncated/corrupt state file (e.g. the process was killed
                # mid-write) must not abort a resume.  Preserve the damaged
                # file for inspection and start from a clean state.
                backup = self.path.with_suffix(".json.corrupt")
                try:
                    self.path.replace(backup)
                    print(f"  [warn] campaign state file was unreadable "
                          f"({exc}); moved to {backup} and starting fresh.")
                except OSError:
                    print(f"  [warn] campaign state file was unreadable "
                          f"({exc}); starting fresh.")
        return {"iterations": {}, "submitted_at": str(datetime.now())}

    # We save the state after every job submission.
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file in the same directory, then atomically
        # rename over the real path.
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # When we submit a job, we record its ID and timestamp as well as iteration and phase. 
    def record_job(self, iteration: int, phase: int, job_id: str):
        key = str(iteration)
        self.data["iterations"].setdefault(key, {})
        self.data["iterations"][key][f"phase{phase}_job_id"] = job_id
        self.data["iterations"][key][f"phase{phase}_submitted"] = str(datetime.now())
        self.save()

    # When resuming, we can look up the last submitted job ID for a given iteration and phase.
    def get_job_id(self, iteration: int, phase: int) -> str | None:
        return (self.data["iterations"]
                .get(str(iteration), {})
                .get(f"phase{phase}_job_id"))

    # This allows us to check if a phase has already been submitted, so we don't resubmit it. 
    def is_phase_submitted(self, iteration: int, phase: int) -> bool:
        return self.get_job_id(iteration, phase) is not None


# =============================================================================
# Scheduler abstraction
# =============================================================================

class Scheduler:
    """
    Thin wrapper around SLURM / PBS / SGE submission commands.
    All scheduler-specific syntax is isolated here - the rest of the code
    is scheduler-agnostic.
    """

    # Max seconds to wait for a submission command to return.  Submission is
    # a quick scheduler RPC. If it hasn't answered by now something is wrong
    # and we should fail loudly.
    SUBMIT_TIMEOUT = 60

    def __init__(self, stype: str, account: str, dry_run: bool = False):
        self.stype = stype.upper() # Convert scheduler type to uppercase for consistency
        self.account = account
        self.dry_run = dry_run
        if self.stype not in ("SLURM", "PBS", "SGE"):
            raise ValueError(f"Unsupported scheduler: {stype}. "
                             f"Choose from: SLURM, PBS, SGE")

    def submit(self, script_path: str, depends_on: str | None = None,
               array: str | None = None) -> str:
        """Submit a script, optionally depending on a previous job ID.
        Returns the new job ID string.

        `array` (e.g. "1-500%20") submits a job array. The returned ID is the
        array's job ID, and an afterok dependency on it waits for every task.

        Raises RuntimeError (with the scheduler's own stderr) on any
        submission failure or timeout.
        """
        cmd = self._build_submit_cmd(script_path, depends_on, array) # Build the appropriate submission command based on scheduler type.
        print(f"  Submitting: {' '.join(cmd)}")

        if self.dry_run:
            fake_id = f"DRY_{Path(script_path).stem}"
            print(f"  [dry-run] Would submit -> fake job ID: {fake_id}")
            return fake_id

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
                timeout=self.SUBMIT_TIMEOUT,
            )  # Execute the submission command, stdout carries the new job ID.
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Scheduler command not found: {cmd[0]!r}. "
                f"Is {self.stype} available on this host / PATH?"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Submission timed out after {self.SUBMIT_TIMEOUT}s: "
                f"{' '.join(cmd)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Submission failed (exit {exc.returncode}) for "
                f"{Path(script_path).name}:\n{(exc.stderr or '').strip()}"
            ) from exc

        job_id = self._parse_job_id(result.stdout.strip())
        print(f"  -> Job ID: {job_id}")
        return job_id

    def _build_submit_cmd(self, script: str, depends_on: str | None,
                          array: str | None = None) -> list[str]:
        """Construct the appropriate submission command based on the scheduler
        type, dependency, and optional array specification."""

        # Builds command used to submit a job script to the scheduler.
        if self.stype == "SLURM":
            cmd = ["sbatch"] # starts the command with the submit tool for SLURM.
            if depends_on: # Checks whether this job should wait for another job first (i.e., if depends_on is not None).
                cmd += [f"--dependency=afterok:{depends_on}"] # Adds a dependency option so this job only runs after the named job succeeds.
            if array:  # e.g. "1-500%20": array of 500 tasks, at most 20 running at once.
                cmd += [f"--array={array}"]
            cmd.append(script) # Adds the script file path to the command.
            return cmd

        if self.stype == "PBS":
            cmd = ["qsub"]
            if depends_on:
                cmd += ["-W", f"depend=afterok:{depends_on}"]
            if array:
                cmd += ["-J", array]           # PBS Pro job array
            cmd.append(script)
            return cmd

        # SGE
        cmd = ["qsub"]
        if depends_on:
            cmd += ["-hold_jid", depends_on]
        if array:
            cmd += ["-t", array]               # SGE array tasks
        cmd.append(script)
        return cmd

    def _parse_job_id(self, stdout: str) -> str:
        """Extract the job ID from the scheduler's submission output.

        Raises RuntimeError if the output does not have the expected shape.
        """
        fields = stdout.split()
        if not fields:
            raise RuntimeError(
                f"{self.stype} submission returned no output; cannot "
                f"determine job ID."
            )
        try:
            if self.stype == "SLURM":
                return fields[-1]                # "Submitted batch job 12345"
            if self.stype == "PBS":
                return stdout.split(".")[0]      # "12345.cluster.name"
            # SGE: 'Your job 12345 ("name") has been submitted'
            return fields[2]
        except IndexError as exc:
            raise RuntimeError(
                f"Could not parse job ID from {self.stype} output: "
                f"{stdout!r}"
            ) from exc

    def header(self, job_name: str, walltime: str, nodes: int,
               cpus: int, mem: str, gpus: int, account: str,
               partition: str, log_dir: str, gpu_type: str = "",
               array_log: bool = False, exclude: str = "") -> str:
        """Return the scheduler-specific resource header for a job script.

        When array_log is True, SLURM output/error filenames use %A_%a
        (array-job id + task id) so each array task logs to its own file.
        `exclude` (e.g. "fc10101,fc10102") keeps jobs off known-bad nodes.
        """
        # SLURM log tag: per-array-task file for arrays, per-job file otherwise.
        slurm_tag = "%A_%a" if array_log else "%j"
        # GPU type is required on some clusters (e.g. Alliance rejects a bare gpu request
        # and demand a model, so --gres=gpu:h100:1 rather than --gres=gpu:1).
        # When gpu_type is set, name the model; when blank, request by count.
        gtype = (gpu_type or "").strip()
        slurm_gres = f"gpu:{gtype}:{gpus}" if gtype else f"gpu:{gpus}"
        gpu_lines = {
            "SLURM": f"#SBATCH --gres={slurm_gres}",
            "PBS":   f"#PBS -l ngpus={gpus}",
            "SGE":   f"#$ -l gpu={gpus}",
        }
        gpu_line = (gpu_lines[self.stype] + "\n") if gpus > 0 else ""

        # Partition/queue is optional: many clusters reject an explicit partition and schedule from the
        # account alone.  When the config leaves it blank, omit the directive.
        part = (partition or "").strip()
        part_lines = {
            "SLURM": f"#SBATCH --partition={part}",
            "PBS":   f"#PBS -q {part}",
            "SGE":   f"#$ -q {part}",
        }
        part_line = (part_lines[self.stype] + "\n") if part else ""

        # Optional node exclusion.
        excl = (exclude or "").strip()
        excl_lines = {
            "SLURM": f"#SBATCH --exclude={excl}",
            "PBS":   "",   # PBS/SGE node exclusion is site-specific. skip.
            "SGE":   "",
        }
        excl_line = (excl_lines[self.stype] + "\n") if (excl and excl_lines[self.stype]) else ""

        if self.stype == "SLURM":
            return textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --account={account}
                #SBATCH --nodes={nodes}
                #SBATCH --cpus-per-task={cpus}
                #SBATCH --mem={mem}
                #SBATCH --time={walltime}
                #SBATCH --output={log_dir}/{job_name}_{slurm_tag}.out
                #SBATCH --error={log_dir}/{job_name}_{slurm_tag}.err
                """) + part_line + gpu_line + excl_line

        if self.stype == "PBS":
            return textwrap.dedent(f"""\
                #!/bin/bash
                #PBS -N {job_name}
                #PBS -A {account}
                #PBS -l nodes={nodes}:ppn={cpus}
                #PBS -l mem={mem}
                #PBS -l walltime={walltime}
                #PBS -o {log_dir}/{job_name}.out
                #PBS -e {log_dir}/{job_name}.err
                """) + part_line + gpu_line

        # SGE
        return textwrap.dedent(f"""\
            #!/bin/bash
            #$ -N {job_name}
            #$ -A {account}
            #$ -pe smp {cpus}
            #$ -l h_vmem={mem}
            #$ -l h_rt={walltime}
            #$ -o {log_dir}/{job_name}.out
            #$ -e {log_dir}/{job_name}.err
            """) + part_line + gpu_line


# =============================================================================
# Job script generators  (one per phase)
# =============================================================================

# Phase metadata used both for submission ordering and status display.
# Defined once at module level so it isn't rebuilt on every run() call.
PHASES = {
    1: "Sampling",
    2: "Ligand prep",
    3: "Docking",
    4: "Training",
    5: "Inference",
}

# Supported open-source docking engines.
SUPPORTED_PROGRAMS = ("GNINA", "AUTODOCK_GPU")


def _docking_program(cfg: dict) -> str:
    """Return the normalised docking program, or raise with a clear message.

    Accepts a couple of spelling variants for AutoDock-GPU for convenience.
    """
    raw = str(cfg["docking"]["program"]).upper().replace("-", "_")
    if raw in ("AUTODOCKGPU", "AUTODOCK_GPU", "ADGPU"):
        return "AUTODOCK_GPU"
    if raw == "GNINA":
        return "GNINA"
    raise ValueError(
        f"Unsupported docking program: {cfg['docking']['program']!r}. "
        f"Choose one of: {', '.join(SUPPORTED_PROGRAMS)}."
    )


class JobScriptFactory:
    """
    Generates the body of each phase's job script.
    All DD command calls mirror the Gentile et al. 2022 protocol exactly,
    with paths and parameters substituted from config.
    """

    def __init__(self, cfg: dict, scheduler: Scheduler):
        self.cfg = cfg
        self.s = scheduler

        # We store the project directory and campaign name for convenience.
        self.proj   = cfg["project_dir"]
        self.name   = cfg["campaign_name"]

        # Directory holding this package's helper scripts (dd_ligand_prep.py,
        # dd_autodock_export.py).
        self.pkg_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # Shared preamble written at the top of every script
    # ------------------------------------------------------------------
    def _preamble(self, iteration: int, gpu: bool = False) -> str:
        dd_dir = self.cfg["env"]["dd_protocol_dir"]

        # How to activate the Python environment inside each job.  The wizard
        # writes env.activate (a conda-activate line, or `source venv/bin/activate`
        # for the no-conda / virtualenv path).  Fall back to conda for older
        # configs that only have env.conda_env.
        activate = self.cfg["env"].get("activate")
        if not activate:
            conda_env = self.cfg["env"].get("conda_env", "")
            activate = ('source "$(conda info --base)/etc/profile.d/conda.sh" '
                        f'&& conda activate "{conda_env}"')

        # Cluster modules that provide the docking engine (gnina / AutoDock-GPU /
        # autogrid4, ...).  These are `module load`ed inside every job.  Guarded with `type module` so
        # the script still runs on systems without an environment-module system.
        modules = self.cfg["env"].get("modules") or []
        module_block = ""
        if modules:
            mod_list = " ".join(modules)
            module_block = (
                "# Load cluster modules that provide the docking tools + toolchain.\n"
                "# A CVMFS/Lmod outage on the compute node makes these fail; abort\n"
                "# loudly.\n"
                "if type module &>/dev/null; then\n"
                f"    module load {mod_list} || {{ echo \"ERROR: 'module load {mod_list}' failed - likely a CVMFS/Lmod problem on this compute node (e.g. 'Transport endpoint is not connected'). Resubmit; if it recurs, report the node to support.\" >&2; exit 1; }}\n"
                "fi\n"
            )

        preamble = textwrap.dedent(f"""\

            # -- Environment setup -----------------------------------------
            # Abort on any failure from here on so the scheduler marks the job
            # FAILED.  `set -u` is deferred until after activation.
            set -eo pipefail

            export DD_PROJECT_DIR="{self.proj}"
            export DD_ITERATION={iteration}
            export DD_CAMPAIGN="{self.name}"
            export DD_PROTOCOL_DIR="{dd_dir}"

            __DD_MODULES__
            # Activate the Python environment (conda env or virtualenv).
            # Modules are loaded first so an Alliance-style venv sees its
            # matching python module, and gnina's prerequisites are in place.
            {activate} || {{ echo "ERROR: failed to activate the Python environment. Check for CVMFS/Lmod errors above." >&2; exit 1; }}

            # Fail fast if the environment did not actually come up,
            # e.g. a CVMFS/Lmod outage left the modules unloaded and we are on
            # the bare system python.
            python -c "import numpy, pandas, rdkit" 2>/dev/null || {{ echo "ERROR: python environment is not usable (numpy/pandas/rdkit import failed). Modules or virtualenv did not activate correctly - check for CVMFS/Lmod errors above." >&2; exit 1; }}

            set -u    # environment is up; now also catch unset variables

            __DD_GPUCHECK__
            # nullglob: an unmatched glob expands to nothing, so `for f in dir/*.smi`
            # never feeds a bogus "dir/*.smi" path into a tool.  Loops that require
            # input guard against emptiness (below).
            shopt -s nullglob

            echo "[$(date)] Starting iteration ${{DD_ITERATION}}"
        """)

        # GPU health check: Fail fast and name the node so this job does
        # not run docking/training/inference on a bad device. The node can then
        # be excluded (scheduler.exclude_nodes or sbatch --exclude).
        gpu_block = ""
        if gpu:
            gpu_block = textwrap.dedent("""\
                if command -v nvidia-smi &>/dev/null; then
                    if ! nvidia-smi >/dev/null 2>&1; then
                        echo "ERROR: nvidia-smi failed on $(hostname). GPU likely faulty (needs reset). Resubmit excluding this node: add it to scheduler.exclude_nodes in campaign.yaml, or use sbatch --exclude=$(hostname)." >&2
                        exit 1
                    fi
                    echo "[$(date)] GPU healthy on $(hostname):"
                    nvidia-smi -L || true
                fi
                """)

        return (preamble
                .replace("__DD_MODULES__\n", module_block)
                .replace("__DD_GPUCHECK__\n", gpu_block))

    # Fallbacks for phase keys a config does not define.
    _DEFAULT_RES = {"nodes": 1, "cpus": 2, "mem": "8G", "gpus": 0}
    _DEFAULT_WT  = "00:30:00"

    def _make_header(self, phase_key: str, job_name: str,
                     partition_key: str, walltime_override: str | None = None,
                     array_log: bool = False) -> str:
        """Build the scheduler header for any phase using config lookups.

        walltime_override lets a caller set a computed walltime instead of the 
        static config value. array_log switches SLURM logs to per-array-task files.
        """
        sched = self.cfg["scheduler"]
        r   = sched["resources"].get(phase_key, self._DEFAULT_RES)
        wt  = walltime_override or sched["walltime"].get(phase_key, self._DEFAULT_WT)
        acc = sched["account"]
        par = sched.get(partition_key, "")     # optional. blank = omit
        gtype = sched.get("gpu_type", "")      # optional. blank = omit model
        excl = sched.get("exclude_nodes", "")  # optional. blank = exclude none
        log = f"{self.proj}/logs"
        return self.s.header(job_name, wt, r["nodes"], r["cpus"],
                             r["mem"], r["gpus"], acc, par, log, gtype,
                             array_log, excl)

    # Phase 1 (sampling) is one job that reads/samples every fingerprint chunk,
    # so its walltime should grow with the number of chunks.
    _PH1_BASE_SEC       = 1800           # 30 min fixed overhead
    _PH1_PER_CHUNK_SEC  = 15             # generous per-chunk count + sample I/O
    _WALLTIME_CAP_SEC   = 24 * 3600      # never auto-request more than 24 h

    @staticmethod
    def _walltime_to_sec(wt: str) -> int:
        """Parse HH:MM:SS or D-HH:MM:SS into seconds (0 if unparseable)."""
        try:
            days = 0
            if "-" in wt:
                d, wt = wt.split("-", 1)
                days = int(d)
            parts = [int(x) for x in wt.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0)
            h, m, s = parts
            return days * 86400 + h * 3600 + m * 60 + s
        except (ValueError, AttributeError):
            return 0

    def _scaled_phase1_walltime(self, n_chunks: int | None) -> str | None:
        """Return a chunk-count-scaled walltime for Phase 1, or None to keep the
        config value.  Uses max(config, estimate) capped at _WALLTIME_CAP_SEC."""
        if not n_chunks or n_chunks <= 0:
            return None
        est = self._PH1_BASE_SEC + n_chunks * self._PH1_PER_CHUNK_SEC
        cfg_sec = self._walltime_to_sec(
            self.cfg["scheduler"]["walltime"].get("phase1_sampling", "00:30:00"))
        chosen = min(max(est, cfg_sec), self._WALLTIME_CAP_SEC)
        if chosen <= cfg_sec:
            return None                      # config already generous enough
        hh, mm, ss = chosen // 3600, (chosen % 3600) // 60, chosen % 60
        wt = f"{hh:02d}:{mm:02d}:{ss:02d}"
        capped = est > self._WALLTIME_CAP_SEC
        note = "  (CAPPED - consider splitting the library or raising the cap)" if capped else ""
        print(f"  [auto] Phase 1 walltime scaled to {wt} for {n_chunks} chunks{note}")
        return wt

    # ------------------------------------------------------------------
    # Phase 1: Random sampling from library (iter 1) or predictions (iter N>1)
    # ------------------------------------------------------------------
    def phase1_sampling(self, iteration: int, n_chunks: int | None = None) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p1_sampling"
        wt_override = self._scaled_phase1_walltime(n_chunks)
        header   = self._make_header("phase1_sampling", job_name, "cpu_partition",
                                     walltime_override=wt_override)

        dd_dir   = self.cfg["env"]["dd_protocol_dir"]
        ncpu     = self.cfg["scheduler"]["resources"]["phase1_sampling"]["cpus"]
        fp_dir   = self.cfg["library"]["fingerprint_dir"]
        smi_dir  = self.cfg["library"]["smiles_dir"]
        train_sz = self.cfg["dd"]["train_size"]
        val_sz   = self.cfg["dd"]["val_size"]

        # In iteration 1 we sample from the full fingerprint library.
        # In subsequent iterations we sample only from the previous iteration's
        # virtual-hit predictions - validation and test sets are frozen after
        # iteration 1 and reused throughout (see paper Section 'Molecular sample size').
        if iteration == 1:
            data_dir     = fp_dir
            tot_sampling = train_sz + 2 * val_sz
        else:
            data_dir     = f"{self.proj}/{self.name}/iteration_{iteration - 1}/morgan_1024_predictions"
            tot_sampling = train_sz   # only augment training; val/test are fixed

        body = textwrap.dedent(f"""\

            # -- Phase 1: Sampling (iteration {iteration}) ------------------
            # Determine how many molecules to sample from each library chunk,
            # then perform the actual random sampling, deduplicate, and extract
            # both Morgan fingerprints and SMILES for the sampled molecules.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            mkdir -p "$ITER_DIR"

            # Step 1a: count molecules per file to reach target sample size
            python "{dd_dir}/scripts_1/molecular_file_count_updated.py" \\
                --project_name "{self.name}" \\
                --n_iteration {iteration} \\
                --data_directory "{data_dir}" \\
                --tot_process {ncpu} \\
                --tot_sampling {tot_sampling}

            # Step 1b: perform the random sampling
            python "{dd_dir}/scripts_1/sampling.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --n_iteration {iteration} \\
                --data_directory "{data_dir}" \\
                --tot_process {ncpu} \\
                --train_size {train_sz} \\
                --val_size {val_sz}

            # Step 1c: remove overlaps between train / val / test sets
            python "{dd_dir}/scripts_1/sanity_check.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --n_iteration {iteration}

            # Step 1d: extract Morgan fingerprints for sampled molecules
            python "{dd_dir}/scripts_1/extracting_morgan.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --n_iteration {iteration} \\
                --morgan_directory "{fp_dir}" \\
                --tot_process {ncpu}

            # Step 1e: extract SMILES for sampled molecules
            python "{dd_dir}/scripts_1/extracting_smiles.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --n_iteration {iteration} \\
                --smile_directory "{smi_dir}" \\
                --tot_process {ncpu}

            echo "[$(date)] Phase 1 complete - iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 2: 3D ligand preparation (RDKit ETKDG + Meeko)
    # ------------------------------------------------------------------
    def phase2_ligand_prep(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p2_ligprep"
        header   = self._make_header("phase2_ligand_prep", job_name, "cpu_partition")
        program  = _docking_program(self.cfg)
        ncpu     = self.cfg["scheduler"]["resources"]["phase2_ligand_prep"]["cpus"]

        # Open-source 3D prep (RDKit ETKDG + optional Meeko), replacing OMEGA.
        #   GNINA        -> 3D SDF per chunk        (Gnina docks multi-mol SDF)
        #   AUTODOCK_GPU -> one PDBQT per molecule  (AutoDock-GPU docks 1/ligand)
        out_format = "sdf" if program == "GNINA" else "pdbqt"

        body = textwrap.dedent(f"""\

            # -- Phase 2: Ligand preparation (iteration {iteration}) --------
            # RDKit embeds a 3D conformer for each sampled molecule and
            # minimises it; Meeko converts to PDBQT when AutoDock-GPU is used.
            # Output: $ITER_DIR/sdf/<chunk>.sdf   (Gnina)
            #     or  $ITER_DIR/pdbqt/<chunk>/<molid>.pdbqt   (AutoDock-GPU)

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"

            python "{self.pkg_dir}/dd_ligand_prep.py" \\
                --smiles-dir "$ITER_DIR/smile" \\
                --out-dir    "$ITER_DIR" \\
                --format     {out_format} \\
                --nprocs     {ncpu}

            echo "[$(date)] Phase 2 complete - iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 3: Docking
    # ------------------------------------------------------------------
    # ---- Phase 3: shard, dock array, merge --------------------------------
    #
    # Docking is the dominant cost of a campaign, so it runs as a job array:
    # 3a splits the prepared ligands into fixed-size shards, 3b docks one shard
    # per array task, 3c concatenates the results back together.

    _SHARD_DIR  = "sdf_shards"
    _DOCKED_DIR = "docked_shards"

    def _shard_count(self, iteration: int) -> int:
        """How many array tasks Phase 3b needs.

        The array size has to be fixed when the job is submitted, but the exact
        molecule count is only known once 3a has run. So it is computed from the
        configured sample sizes and rounded up, with a margin: ligand prep drops
        a few molecules that fail 3-D embedding, but nothing can ever produce
        more than was sampled, so over-provisioning is the safe direction.
        Surplus tasks find no shard file and exit 0 without doing anything.
        """
        dd = self.cfg["dd"]
        per_job = int(self.cfg["docking"].get("molecules_per_docking_job", 10000))
        if per_job < 1:
            raise ValueError("docking.molecules_per_docking_job must be >= 1")
        # Iteration 1 samples train + validation + test; later iterations only
        # draw a fresh training batch.
        if iteration == 1:
            total = int(dd["train_size"]) + 2 * int(dd["val_size"])
        else:
            total = int(dd["train_size"])
        return max(1, -(-total // per_job))     # ceiling division

    # The two helper programs below are plain strings not f-strings. They are
    # Python source embedded in a heredoc, and f-string interpolation would eat
    # every {name} and {idx:05d} in them.
    _SPLIT_PY = r"""
import os, sys

shard_dir, per_job = sys.argv[1], int(sys.argv[2])
inputs = sys.argv[3:]
total_shards = 0

for path in inputs:
    base = os.path.basename(path)[:-4]        # strip ".sdf"
    n_mol = shard_idx = 0
    out = None
    # SDF records end with a line that is exactly "$$$$". Splitting on that
    # boundary keeps every record intact; splitting on byte offsets or line
    # counts would cut molecules in half.
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if out is None:
                shard_idx += 1
                total_shards += 1
                out = open(os.path.join(
                    shard_dir, "%s__%05d.sdf" % (base, shard_idx)), "w")
            out.write(line)
            if line.rstrip("\n").rstrip("\r") == "$$$$":
                n_mol += 1
                if n_mol % per_job == 0:
                    out.close()
                    out = None
    if out is not None:
        out.close()
    print("  %s: %d molecules -> %d shard(s)" % (base, n_mol, shard_idx),
          flush=True)

print("TOTAL_SHARDS=%d" % total_shards)
with open(os.path.join(shard_dir, ".n_shards"), "w") as fh:
    fh.write(str(total_shards))
"""

    _MERGE_PY = r"""
import os, re, sys
from collections import defaultdict

shard_out, docked = sys.argv[1], sys.argv[2]
groups = defaultdict(list)
pat = re.compile(r"^(?P<base>.+)__(?P<idx>[0-9]{5})_docked\.sdf$")

for name in os.listdir(shard_out):
    m = pat.match(name)
    if m:
        groups[m.group("base")].append((int(m.group("idx")), name))

if not groups:
    sys.exit("ERROR: no docked shards found in %s" % shard_out)

for base, items in sorted(groups.items()):
    items.sort()                       # shard order, so output is deterministic
    out_path = os.path.join(docked, "%s_docked.sdf" % base)
    n_mol = 0
    with open(out_path, "w") as out:
        for _, name in items:
            with open(os.path.join(shard_out, name), "r", errors="replace") as fh:
                for line in fh:
                    out.write(line)
                    if line.rstrip("\n").rstrip("\r") == "$$$$":
                        n_mol += 1
    print("  %s: %d shard(s) -> %d molecules" % (base, len(items), n_mol),
          flush=True)

print("MERGED_SETS=%d" % len(groups))
"""

    def phase3a_split(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p3a_split"
        header   = self._make_header("phase3a_split", job_name, "cpu_partition")
        per_job  = int(self.cfg["docking"].get("molecules_per_docking_job", 10000))

        body = textwrap.dedent(f"""\

            # -- Phase 3a: split prepared ligands into shards (iteration {iteration}) --
            # Each input SDF becomes <base>__NNNNN.sdf shards of {per_job}
            # molecules. The base name is preserved so 3c can group shards back
            # by set (train / valid / test).

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            SHARD_DIR="$ITER_DIR/{self._SHARD_DIR}"
            rm -rf "$SHARD_DIR"
            mkdir -p "$SHARD_DIR"

            SDF_FILES=("$ITER_DIR/sdf/"*.sdf)
            if [ ! -e "${{SDF_FILES[0]}}" ]; then
                echo "ERROR: no .sdf files in $ITER_DIR/sdf/ - Phase 2 output missing" >&2
                exit 1
            fi

            python - "$SHARD_DIR" {per_job} "${{SDF_FILES[@]}}" <<'PYSPLIT'
{self._SPLIT_PY}
PYSPLIT

            N=$(cat "$SHARD_DIR/.n_shards")
            echo "[$(date)] Phase 3a complete - $N shard(s) written to $SHARD_DIR"
            if [ "$N" -eq 0 ]; then
                echo "ERROR: splitting produced no shards" >&2
                exit 1
            fi
        """)
        return header + self._preamble(iteration) + body

    def phase3b_array(self, iteration: int) -> str:
        """One array task per shard. SLURM_ARRAY_TASK_ID selects the shard."""
        job_name = f"{self.name}_i{iteration:02d}_p3b_dock"
        header   = self._make_header("phase3_docking", job_name, "gpu_partition",
                                     array_log=True)
        program  = _docking_program(self.cfg)
        dock     = self.cfg["docking"]

        if program == "GNINA":
            docking_cmd = self._gnina_docking_cmd(dock)
        else:
            docking_cmd = self._autodock_docking_cmd(dock)

        body = textwrap.dedent(f"""\

            # -- Phase 3b: dock one shard (iteration {iteration}) ---------------
            # Array task N docks the Nth shard. The array is sized from the
            # configured sample sizes, so it may be slightly larger than the
            # number of shards that actually exist. Surplus tasks exit 0.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            SHARD_DIR="$ITER_DIR/{self._SHARD_DIR}"
            OUT_DIR="$ITER_DIR/{self._DOCKED_DIR}"
            mkdir -p "$OUT_DIR"

            mapfile -t SHARDS < <(ls -1 "$SHARD_DIR"/*.sdf 2>/dev/null | sort)
            IDX=$((SLURM_ARRAY_TASK_ID - 1))

            if [ "$IDX" -ge "${{#SHARDS[@]}}" ]; then
                echo "No shard for array index $SLURM_ARRAY_TASK_ID " \
                     "(${{#SHARDS[@]}} shards exist) - nothing to do."
                exit 0
            fi

            SDF_FILE="${{SHARDS[$IDX]}}"
            BASE=$(basename "$SDF_FILE" .sdf)
            OUT_FILE="$OUT_DIR/${{BASE}}_docked.sdf"

            # Resume: a completed shard is left alone, so a resubmitted array
            # only redoes the tasks that did not finish.
            if [ -s "$OUT_FILE" ]; then
                echo "$OUT_FILE already present - skipping."
                exit 0
            fi

            echo "[$(date)] Docking shard $SLURM_ARRAY_TASK_ID: $BASE"
        """) + docking_cmd + textwrap.dedent(f"""\

            echo "[$(date)] Phase 3b task $SLURM_ARRAY_TASK_ID complete"
        """)

        return header + self._preamble(iteration, gpu=True) + body

    def phase3c_merge(self, iteration: int) -> str:
        """Concatenate docked shards back into one SDF per sampled set.

        Restores the filenames Phase 4a expects, so extract_labels.py sees the
        same layout it would have from a single-job Phase 3.
        """
        job_name = f"{self.name}_i{iteration:02d}_p3c_merge"
        header   = self._make_header("phase3c_merge", job_name, "cpu_partition")
        n_expect = 3 if iteration == 1 else 1

        body = textwrap.dedent(f"""\

            # -- Phase 3c: merge docked shards (iteration {iteration}) ----------
            # Shards are named <base>__NNNNN_docked.sdf; grouping on the "__"
            # separator reassembles each set into <base>_docked.sdf.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            SHARD_OUT="$ITER_DIR/{self._DOCKED_DIR}"
            DOCKED="$ITER_DIR/docked"
            mkdir -p "$DOCKED"

            python - "$SHARD_OUT" "$DOCKED" <<'PYMERGE'
{self._MERGE_PY}
PYMERGE

            NFILES=$(ls -1 "$DOCKED"/*_docked.sdf 2>/dev/null | wc -l)
            echo "[$(date)] Phase 3c complete. $NFILES merged file(s) in $DOCKED"

            # Phase 4a passes --tot_process {n_expect}, so a mismatch here means
            # label extraction would read the wrong number of sets.
            if [ "$NFILES" -ne {n_expect} ]; then
                echo "ERROR: expected {n_expect} merged docked file(s), found $NFILES" >&2
                exit 1
            fi
        """)
        return header + self._preamble(iteration) + body

    def _gnina_docking_cmd(self, dock: dict) -> str:
        """Dock one shard ($SDF_FILE -> $OUT_FILE) into the pre-computed box.

        Both variables are set by the Phase 3b array wrapper. The box is read at
        runtime from receptor_box.json, which dd_receptor_prep.py wrote once.
        """
        receptor = dock["receptor_file"]
        box_json = dock["box_json"]
        cnn      = dock.get("gnina_cnn", "rescore")
        exhaust  = dock.get("exhaustiveness", 8)
        # Give gnina the cores the job actually reserved. Without --cpu it uses
        # its own default thread count, which may be fewer than allocated.
        ncpu = self.cfg["scheduler"]["resources"].get(
            "phase3_docking", self._DEFAULT_RES)["cpus"]
        return textwrap.dedent(f"""\
            # Read the pre-computed binding box (center/size) once.
            BOX=$(python -c "import json; d=json.load(open('{box_json}')); \\
c=d['center']; s=d['size']; print(c[0], c[1], c[2], s[0], s[1], s[2])")
            read CX CY CZ SX SY SZ <<< "$BOX"

            # Write to a temporary file and move it into place only on success.
            # The resume check in the array wrapper treats a non-empty output as
            # a finished shard, so a partial file from a killed task would
            # otherwise be mistaken for completed work.
            #
            # The staged name must keep the .sdf extension. gnina picks its
            # output format from the extension, and anything it does not
            # recognise makes it discard every docked pose while still appearing to run.
            TMP_OUT="${{OUT_FILE%.sdf}}.partial.sdf"
            rm -f "$TMP_OUT"

            gnina \\
                --receptor "{receptor}" \\
                --ligand "$SDF_FILE" \\
                --center_x "$CX" --center_y "$CY" --center_z "$CZ" \\
                --size_x "$SX" --size_y "$SY" --size_z "$SZ" \\
                --cnn_scoring {cnn} \\
                --exhaustiveness {exhaust} \\
                --num_modes 1 \\
                --cpu {ncpu} \\
                --seed 42 \\
                --quiet \\
                --out "$TMP_OUT"

            mv "$TMP_OUT" "$OUT_FILE"
        """)

    def _autodock_docking_cmd(self, dock: dict) -> str:
        """AutoDock-GPU docking: batch each chunk's per-molecule PDBQTs against
        the pre-computed grid maps, then export each chunk's .dlg results into a
        single scored SDF for Phase 4.

        NOTE: Phase 3 now shards the prepared ligands and docks one shard per
        array task, but that sharding operates on SDF records. AutoDock-GPU
        consumes a directory of per-molecule PDBQT files instead, so it needs
        its own sharding scheme (by chunk directory) that has not been written
        yet. _submit_phase3 rejects AUTODOCK_GPU rather than generating a script
        that would silently dock the wrong thing."""
        maps_fld = dock["maps_fld"]
        adbin    = dock.get("autodock_bin", "autodock_gpu_128wi")
        nrun     = dock.get("autodock_nrun", 10)
        return textwrap.dedent(f"""\
            # Dock each chunk's per-molecule PDBQTs with AutoDock-GPU (batch mode),
            # then convert the .dlg results to a scored SDF for label extraction.
            CHUNK_DIRS=("$ITER_DIR/pdbqt/"*/)
            if [ ${{#CHUNK_DIRS[@]}} -eq 0 ]; then
                echo "ERROR: no per-chunk pdbqt dirs in $ITER_DIR/pdbqt/ - Phase 2 output missing" >&2
                exit 1
            fi
            for CHUNK_DIR in "${{CHUNK_DIRS[@]}}"; do
                CHUNK=$(basename "$CHUNK_DIR")
                mkdir -p "$ITER_DIR/docked/$CHUNK"

                # Build the AutoDock-GPU batch file: shared maps on line 1, then
                # (ligand pdbqt, result basename) pairs for every molecule.
                # A chunk can hold ~1M ligands, so stream with find (no giant
                # bash array) and use parameter expansion (no per-file forks),
                # writing the batch file in a single open.
                BATCH="$ITER_DIR/docked/$CHUNK.filelist"
                {{
                    echo "{maps_fld}"
                    find "$CHUNK_DIR" -maxdepth 1 -name '*.pdbqt' | while IFS= read -r LIG; do
                        LIGBASE="${{LIG##*/}}"; LIGBASE="${{LIGBASE%.pdbqt}}"
                        printf '%s\\n%s\\n' "$LIG" "$ITER_DIR/docked/$CHUNK/$LIGBASE"
                    done
                }} > "$BATCH"
                if [ "$(wc -l < "$BATCH")" -le 1 ]; then
                    echo "WARNING: no pdbqt ligands in $CHUNK_DIR - skipping" >&2
                    continue
                fi

                {adbin} --filelist "$BATCH" --nrun {nrun}

                # Collapse this chunk's .dlg results into one scored SDF.
                python "{self.pkg_dir}/dd_autodock_export.py" \\
                    --dlg-dir "$ITER_DIR/docked/$CHUNK" \\
                    --out-sdf "$ITER_DIR/docked/${{CHUNK}}_docked.sdf"
            done
        """)

    # ------------------------------------------------------------------
    # Phase 4: DNN model training (split: 4a generate, 4b array, 4c evaluate)
    # ------------------------------------------------------------------
    # DD trains many hyperparameter models per iteration. The reference protocol
    # runs them as PARALLEL jobs; running them sequentially in one job makes the
    # walltime the SUM of all models. So mirror Phase 5: 4a generates one
    # training script per model, 4b runs them as a job ARRAY (one model/task),
    # and 4c picks the best model once they are all trained.
    @staticmethod
    def _num_training_models(nhp: int) -> int:
        """Number of models DD's simple_job_models_manual.py actually generates
        for a given --number_of_hyp (it quantizes to 16/24/48/72/144). Mirrors
        that script's nested-loop sizing so the training array can be sized
        without running the generator first."""
        oss = 3 if nhp >= 72 else (2 if nhp >= 48 else 1)
        bs  = 2 if nhp >= 144 else 1
        nu  = 3 if nhp >= 24 else 2
        return oss * bs * nu * 8      # dropout(2) * bin_array(2) * wt(2) = 8

    def phase4a_labels(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p4a_labels"
        header   = self._make_header("phase4a_labels", job_name, "cpu_partition")

        dd_dir     = self.cfg["env"]["dd_protocol_dir"]
        fp_dir     = self.cfg["library"]["fingerprint_dir"]
        score_kw   = self.cfg["docking"]["score_keyword"]
        total_iter = self.cfg["dd"]["total_iterations"]
        num_models = self.cfg["dd"]["num_models"]
        val_sz     = self.cfg["dd"]["val_size"]
        pct_first  = self.cfg["dd"]["percent_first"]
        pct_last   = self.cfg["dd"]["percent_last"]
        recall     = self.cfg["dd"]["recall"]
        # Python bool -> "True"/"False", which the DD script expects.
        is_last    = str(iteration == total_iter)
        # Iter 1 docks train+val+test (3 SDFs); later iters only the aug batch.
        n_docking_files = 3 if iteration == 1 else 1

        body = textwrap.dedent(f"""\

            # -- Phase 4a: labels + generate one training script per model ------
            # extract_labels turns docking scores into binary hit/non-hit labels;
            # simple_job_models_manual writes simple_job_1.sh .. simple_job_N.sh
            # (one per hyperparameter model). Those scripts `cd $(pwd)` then run a
            # RELATIVE progressive_docking.py, so generate them from scripts_2.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            mkdir -p "$ITER_DIR"

            python "{dd_dir}/scripts_2/extract_labels.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --iteration_no {iteration} \\
                --tot_process {n_docking_files} \\
                --score_keyword '{score_kw}'

            cd "{dd_dir}/scripts_2"
            python "{dd_dir}/scripts_2/simple_job_models_manual.py" \\
                --iteration_no {iteration} \\
                --morgan_directory "{fp_dir}" \\
                --file_path "{self.proj}/{self.name}" \\
                --number_of_hyp {num_models} \\
                --total_iterations {total_iter} \\
                --is_last {is_last} \\
                --number_mol {val_sz} \\
                --percent_first_mols {pct_first} \\
                --percent_last_mols {pct_last} \\
                --recall {recall}

            NMODELS=$(ls "$ITER_DIR/simple_job/"simple_job_*.sh 2>/dev/null | wc -l)
            echo "[$(date)] Phase 4a: generated $NMODELS model-training scripts (iteration {iteration})"
            if [ "$NMODELS" -eq 0 ]; then
                echo "ERROR: simple_job_models_manual.py generated no scripts" >&2
                exit 1
            fi
        """)

        return header + self._preamble(iteration) + body

    def phase4b_array(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p4b_train"
        # Per-task = one model, so this header's walltime is PER MODEL (DD docs:
        # usually <= ~12 h per model).
        header   = self._make_header("phase4_training", job_name, "gpu_partition",
                                     array_log=True)

        body = textwrap.dedent(f"""\

            # -- Phase 4b: train one DNN model per array task (iteration {iteration}) --
            # Array task k trains the k-th hyperparameter model. Work and memory
            # per task are one model's, so models train in parallel and the phase
            # walltime is per MODEL, not the sum. Models are saved to all_models/
            # for 4c to evaluate.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            TID="${{SLURM_ARRAY_TASK_ID:-}}"
            if [ -z "$TID" ]; then
                echo "ERROR: Phase 4b must run as a SLURM job array (SLURM_ARRAY_TASK_ID unset)." >&2
                exit 1
            fi
            SCRIPT="$ITER_DIR/simple_job/simple_job_${{TID}}.sh"
            if [ ! -f "$SCRIPT" ]; then
                echo "ERROR: training script not found for array task $TID: $SCRIPT" >&2
                exit 1
            fi
            echo "[$(date)] Phase 4b task $TID -> $SCRIPT"
            bash "$SCRIPT"
            echo "[$(date)] Phase 4b task $TID complete"
        """)

        return header + self._preamble(iteration, gpu=True) + body

    def phase4c_eval(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p4c_eval"
        # Grid-search evaluation of the trained models. Runs on CPU (TF falls
        # back to CPU) so it doesn't compete for scarce GPUs; it's modest work.
        header   = self._make_header("phase4c_eval", job_name, "cpu_partition")

        dd_dir = self.cfg["env"]["dd_protocol_dir"]
        fp_dir = self.cfg["library"]["fingerprint_dir"]
        val_sz = self.cfg["dd"]["val_size"]
        recall = self.cfg["dd"]["recall"]

        body = textwrap.dedent(f"""\

            # -- Phase 4c: grid search - pick the best model (iteration {iteration}) --
            # Runs after every 4b task; evaluates the trained models and selects
            # the most precise one, which Phase 5 uses for library-wide inference.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"

            python "{dd_dir}/scripts_2/hyperparameter_result_evaluation.py" \\
                --n_iteration {iteration} \\
                --data_path "{self.proj}/{self.name}" \\
                --morgan_directory "{fp_dir}" \\
                --number_mol {val_sz} \\
                --recall {recall}

            echo "[$(date)] Phase 4 complete - iteration {iteration}"
            echo "Best model stats:"
            cat "$ITER_DIR/best_model_stats.txt" 2>/dev/null || true
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 5: Inference over the full library (5a generate, 5b array)
    # ------------------------------------------------------------------
    # DD scores the whole library every iteration, so this phase
    # grows linearly with library size.  Instead of one job looping over
    # every chunk, we generate one inference script per fingerprint chunk (5a) and run them as a Slurm job
    # array (5b). This way, we have one short, constant-memory task per chunk.  Throughput scales
    # by adding tasks so we don't lengthen job time.
    def phase5a_generate(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p5a_predgen"
        header   = self._make_header("phase5a_predgen", job_name, "cpu_partition")

        dd_dir = self.cfg["env"]["dd_protocol_dir"]
        fp_dir = self.cfg["library"]["fingerprint_dir"]

        body = textwrap.dedent(f"""\

            # -- Phase 5a: generate one inference script per fingerprint chunk --
            # simple_job_predictions_manual.py writes simple_job_1.sh ..
            # simple_job_N.sh (one per fingerprint file).  Each of those scripts
            # does `cd $(pwd)` then runs a relative Prediction_morgan_1024.py, so
            # we must generate them from scripts_2 for that path to resolve on
            # the compute node when Phase 5b runs them.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            mkdir -p "$ITER_DIR"

            cd "{dd_dir}/scripts_2"
            python "{dd_dir}/scripts_2/simple_job_predictions_manual.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --n_iteration {iteration} \\
                --morgan_directory "{fp_dir}"

            NSCRIPTS=$(ls "$ITER_DIR/simple_job_predictions/"simple_job_*.sh 2>/dev/null | wc -l)
            echo "[$(date)] Phase 5a: generated $NSCRIPTS inference scripts (iteration {iteration})"
            if [ "$NSCRIPTS" -eq 0 ]; then
                echo "ERROR: simple_job_predictions_manual.py generated no scripts" >&2
                exit 1
            fi
        """)

        return header + self._preamble(iteration) + body

    def phase5b_array(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p5b_infer"
        # Reuse the phase5_inference GPU resource spec, but PER TASK: each task
        # scores a single chunk, so the walltime/mem there is a generous cap.
        header   = self._make_header("phase5_inference", job_name, "gpu_partition",
                                     array_log=True)

        body = textwrap.dedent(f"""\

            # -- Phase 5b: library-wide inference (job ARRAY, iteration {iteration}) --
            # Array task k scores fingerprint chunk k with the Phase-4 model and
            # writes survivors to morgan_1024_predictions/.  The array width
            # (1-N) is set at submission from the fingerprint-chunk count.  Work
            # and memory per task are constant.

            ITER_DIR="{self.proj}/{self.name}/iteration_{iteration}"
            TID="${{SLURM_ARRAY_TASK_ID:-}}"
            if [ -z "$TID" ]; then
                echo "ERROR: Phase 5b must run as a SLURM job array (SLURM_ARRAY_TASK_ID unset)." >&2
                exit 1
            fi
            SCRIPT="$ITER_DIR/simple_job_predictions/simple_job_${{TID}}.sh"
            if [ ! -f "$SCRIPT" ]; then
                echo "ERROR: inference script not found for array task $TID: $SCRIPT" >&2
                exit 1
            fi
            echo "[$(date)] Phase 5b task $TID -> $SCRIPT"
            bash "$SCRIPT"
            echo "[$(date)] Phase 5b task $TID complete"
        """)

        return header + self._preamble(iteration, gpu=True) + body

    # ------------------------------------------------------------------
    # Final phase: extract SMILES of surviving virtual hits for final docking
    # ------------------------------------------------------------------
    def final_extraction(self, last_iteration: int) -> str:
        job_name = f"{self.name}_final_extraction"
        header   = self._make_header("final_extraction", job_name, "cpu_partition")

        dd_dir   = self.cfg["env"]["dd_protocol_dir"]
        smi_dir  = self.cfg["library"]["smiles_dir"]
        ncpu     = self.cfg["scheduler"]["resources"]["final_extraction"]["cpus"]

        body = textwrap.dedent(f"""\

            # -- Final extraction: retrieve SMILES for all surviving virtual hits --
            # After the last DD iteration, the morgan_1024_predictions folder
            # contains IDs of molecules the DNN predicts are top-scorers.
            # This step maps those IDs back to SMILES so they can be prepared
            # for final explicit docking.

            LAST_PRED="{self.proj}/{self.name}/iteration_{last_iteration}/morgan_1024_predictions"

            python "{dd_dir}/utilities/final_extraction.py" \\
                -smile_dir "{smi_dir}" \\
                -prediction_dir "$LAST_PRED" \\
                -processors {ncpu}

            echo "[$(date)] Final extraction complete."
            echo "Output: smiles.csv and id_score.csv in the current directory."
            echo "These molecules are ready for final 3D preparation and docking."
        """)

        return header + self._preamble(last_iteration) + body


# =============================================================================
# Orchestrator
# =============================================================================

# Aggregation priority for collapsing many task states into one: an incomplete/bad task dominates,
# so a phase only counts as COMPLETED when every row is COMPLETED.
_STATE_PRIORITY = [
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
    "CANCELLED", "REVOKED", "PREEMPTED", "SUSPENDED",
    "RUNNING", "REQUEUED", "PENDING", "COMPLETED",
]


def _aggregate_states(states: list[str]) -> str:
    norm = [s.split("+")[0].strip().upper() for s in states if s.strip()]
    if not norm:
        return "MISSING"
    for st in _STATE_PRIORITY:
        if st in norm:
            return st
    return norm[0]


class DDOrchestrator:
    """
    Builds and submits the full DD active-learning chain.
    Each phase is chained to the previous one via scheduler dependencies,
    so no polling or cron is needed.
    """

    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg        = cfg
        self.proj       = cfg["project_dir"]
        self.dry_run    = dry_run
        self.total_iter = cfg["dd"]["total_iterations"]
        self.scheduler  = Scheduler(cfg["scheduler"]["type"],
                                    cfg["scheduler"]["account"],
                                    dry_run)
        self.factory    = JobScriptFactory(cfg, self.scheduler)
        self.state      = CampaignState(self.proj)
        self.scripts_dir = Path(self.proj) / "job_scripts"
        self.log_dir     = Path(self.proj) / "logs"
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.scripts_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _write_script(self, name: str, content: str) -> str:
        """Write a job script to disk and return its path."""
        path = self.scripts_dir / f"{name}.sh"
        path.write_text(content)
        path.chmod(0o755)
        return str(path)

    def _submit_phase(self, iteration: int, phase, script_content: str,
                      depends_on: str | None, array: str | None = None) -> str:
        """Write the script, record it, and submit it.

        `phase` may be an int (1-4) or a string sub-key ("5a"/"5b").
        `array` (e.g. "1-500%20") submits the script as a job array.
        """
        script_name = f"iter_{iteration:02d}_phase{phase}"
        path = self._write_script(script_name, script_content)

        if self.state.is_phase_submitted(iteration, phase):
            existing = self.state.get_job_id(iteration, phase)
            print(f"  [skip] iter {iteration} phase {phase} already submitted "
              f"(job {existing}) - using existing ID for dependency chain")
            return existing

        job_id = self.scheduler.submit(path, depends_on, array=array)
        # Never persist dry-run job IDs: a later real run would
        # see the phase as already submitted and reuse the fake ID as a real
        # scheduler dependency, which the scheduler rejects.
        if not self.dry_run:
            self.state.record_job(iteration, phase, job_id)
        return job_id

    def _count_fp_chunks(self) -> int:
        """Count fingerprint chunk files (*.txt) in the library.  This is the
        number of Phase 5 inference tasks and drives Phase 1 walltime scaling.
        Environment variables ($SCRATCH, ...) are expanded.  Returns 0 if the
        directory can't be read."""
        fp_dir = os.path.expandvars(self.cfg["library"]["fingerprint_dir"])
        try:
            return sum(1 for _ in Path(fp_dir).glob("*.txt"))
        except OSError:
            return 0

    def _submit_phase3(self, iteration: int, depends_on: str | None,
                       throttle: int, resume: bool = False) -> str:
        """Submit Phase 3 as three chained jobs: 3a shards the prepared ligands,
        3b docks one shard per array task, 3c merges the results back into the
        per-set docked SDFs Phase 4a expects. Returns the 3c job ID.

        Resume handling follows _submit_phase4/_submit_phase5: a sub-step that
        already COMPLETED is skipped, and the next step must not afterok-depend
        on a job that may have been purged, so it depends on None instead.
        Its inputs are already on disk.
        """
        program = _docking_program(self.cfg)
        if program != "GNINA":
            raise SystemExit(
                f"ERROR: Phase 3 array docking is implemented for GNINA only. "
                f"docking.program is {program}. AutoDock-GPU consumes per-molecule "
                f"PDBQT directories rather than SDF records and needs its own "
                f"sharding scheme."
            )

        n_shards = self.factory._shard_count(iteration)
        per_job = self.cfg["docking"].get("molecules_per_docking_job", 10000)
        print(f"  Phase 3: Docking  (split + array over ~{n_shards} shards of "
              f"{per_job} molecules, <= {throttle} concurrent tasks) + merge")

        a_already = self.state.is_phase_submitted(iteration, "3a")
        split_id = self._submit_phase(iteration, "3a",
                                      self.factory.phase3a_split(iteration),
                                      depends_on)

        b_already = self.state.is_phase_submitted(iteration, "3b")
        b_dep = None if (resume and a_already) else split_id
        arr_id = self._submit_phase(iteration, "3b",
                                    self.factory.phase3b_array(iteration),
                                    b_dep, array=f"1-{n_shards}%{throttle}")

        c_dep = None if (resume and b_already) else arr_id
        return self._submit_phase(iteration, "3c",
                                  self.factory.phase3c_merge(iteration), c_dep)

    def _submit_phase5(self, iteration: int, depends_on: str | None,
                       n_chunks: int, throttle: int, resume: bool = False) -> str:
        """Submit Phase 5 as two chained jobs. 5a generates one inference script
        per fingerprint chunk, and 5b runs them as a job array (1-N%throttle).
        Returns the array job ID (the dependency for whatever runs next).

        In resume mode, if 5a is already COMPLETED (kept in state, skipped here)
        its job may be purged, so 5b must not afterok-depend on it; the
        generated scripts are already on disk, so 5b depends on nothing."""
        print(f"  Phase 5: Inference  (generate + array over {n_chunks} chunks, "
              f"<= {throttle} concurrent tasks)")
        a_already = self.state.is_phase_submitted(iteration, "5a")
        gen_script = self.factory.phase5a_generate(iteration)
        gen_id = self._submit_phase(iteration, "5a", gen_script, depends_on)

        # 5b depends on 5a only if 5a was actually (re)submitted this run.
        b_dep = None if (resume and a_already) else gen_id
        arr_script = self.factory.phase5b_array(iteration)
        array_spec = f"1-{n_chunks}%{throttle}"
        return self._submit_phase(iteration, "5b", arr_script, b_dep,
                                  array=array_spec)

    def _submit_phase4(self, iteration: int, depends_on: str | None,
                       throttle: int, resume: bool = False) -> str:
        """Submit Phase 4 as three chained jobs: 4a generates one training
        script per hyperparameter model, 4b trains them as a job array
        (1-N%throttle), and 4c evaluates them and picks the best. Returns the
        4c job ID (the dependency for Phase 5).

        As in Phase 5, when resuming, a sub-step that is already COMPLETED
        (kept, skipped here) must not be used as an afterok dependency for the
        next sub-step. Its inputs are on disk, so that step depends on None."""
        n_models = self.factory._num_training_models(self.cfg["dd"]["num_models"])
        print(f"  Phase 4: Training  (generate + array over {n_models} models, "
              f"<= {throttle} concurrent) + evaluate")

        a_already = self.state.is_phase_submitted(iteration, "4a")
        gen_id = self._submit_phase(iteration, "4a",
                                    self.factory.phase4a_labels(iteration),
                                    depends_on)

        b_already = self.state.is_phase_submitted(iteration, "4b")
        b_dep = None if (resume and a_already) else gen_id
        arr_id = self._submit_phase(iteration, "4b",
                                    self.factory.phase4b_array(iteration),
                                    b_dep, array=f"1-{n_models}%{throttle}")

        c_dep = None if (resume and b_already) else arr_id
        return self._submit_phase(iteration, "4c",
                                  self.factory.phase4c_eval(iteration), c_dep)

    def _actual_state(self, job_id: str | None) -> str:
        """Query the scheduler for a job's aggregated final state (array-aware).
        Returns COMPLETED / CANCELLED / FAILED / ... or MISSING/UNKNOWN."""
        if not job_id:
            return "MISSING"
        stype = self.cfg["scheduler"]["type"]
        try:
            if stype == "SLURM":
                r = subprocess.run(
                    ["sacct", "-j", str(job_id), "--format=State",
                     "--noheader", "-P"],
                    capture_output=True, text=True, timeout=30)
                states = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                return _aggregate_states(states)
            # PBS/SGE: no reliable historical lookup here so we treat as UNKNOWN so
            # the phase is resubmitted (re-running a done phase overwrites its
            # outputs, so this is safe if occasionally redundant).
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "UNKNOWN"
        return "UNKNOWN"

    def _prepare_resume(self) -> tuple[int, int] | None:
        """For --resume: find the first phase whose real scheduler state is not
        COMPLETED, delete it (and everything after it) from the state ledger so
        it will be resubmitted, and return (start_iter, start_phase).  Completed
        phases are kept and never re-run.  Returns None if nothing needs redoing.
        """
        order: list[tuple] = []
        for it in range(1, self.total_iter + 1):
            for ph in (1, 2, "3a", "3b", "3c",
                       "4a", "4b", "4c", "5a", "5b"):
                order.append((it, ph))
        order.append(("final", "final"))

        resume_idx = None
        for idx, (it, ph) in enumerate(order):
            if ph == "final":
                jid = self.state.data.get("final_extraction_job_id")
            else:
                jid = self.state.get_job_id(it, ph)
            st = self._actual_state(jid)
            if st != "COMPLETED":
                resume_idx = idx
                where = "final extraction" if ph == "final" else f"iter {it} phase {ph}"
                shown = st if jid else "not submitted"
                print(f"  Resume point: {where}  (state: {shown})")
                break

        if resume_idx is None:
            return None

        # Clear the resume step and everything after it from the ledger.
        for it, ph in order[resume_idx:]:
            if ph == "final":
                self.state.data.pop("final_extraction_job_id", None)
            else:
                itd = self.state.data["iterations"].get(str(it))
                if itd:
                    itd.pop(f"phase{ph}_job_id", None)
                    itd.pop(f"phase{ph}_submitted", None)
        self.state.save()

        it, ph = order[resume_idx]
        if ph == "final":
            return (self.total_iter + 1, 1)     # loop is empty; only final runs
        if ph in ("3a", "3b", "3c"):
            return (it, 3)
        if ph in ("4a", "4b", "4c"):
            return (it, 4)
        if ph in ("5a", "5b"):
            return (it, 5)
        return (it, ph)

    def run(self, start_iter: int = 1, start_phase: int = 1,
            resume: bool = False):
        """
        Submit the full DD campaign.
        Each iteration submits phases 1-5 in a dependency chain.
        The final extraction is submitted after the last iteration's phase 5.

        resume=True inspects the scheduler for each recorded phase's actual
        state, keeps the COMPLETED ones, and resubmits from the first incomplete
        phase with a fresh dependency chain (so it never depends on a cancelled
        or purged job).
        """
        print(f"\n{'='*60}")
        print(f"  Deep Docking Campaign: {self.cfg['campaign_name']}")
        print(f"  Total iterations: {self.total_iter}")
        print(f"  Scheduler: {self.cfg['scheduler']['type']}")
        print(f"  Project dir: {self.proj}")
        print(f"{'='*60}\n")

        last_job_id = None
        resume_mode = False

        if resume:
            print("Resume: checking actual scheduler state of each phase...")
            rp = self._prepare_resume()
            if rp is None:
                print("Nothing to resume. Every recorded phase already "
                      "COMPLETED (campaign finished).\n")
                return
            start_iter, start_phase = rp
            resume_mode = True
            # Deliberately DO NOT chain onto prior (completed) jobs: their inputs
            # are already on disk, and afterok on a purged job id fails.
            print(f"Resuming from iteration {start_iter}, phase {start_phase} "
                  f"(completed phases kept; fresh dependency chain).\n")
        elif start_iter > 1 or start_phase > 1:
            last_job_id = self._find_resume_job_id(start_iter, start_phase)
            print(f"Resuming from iteration {start_iter}, phase {start_phase}")
            print(f"Chaining from job ID: {last_job_id}\n")

        # Count fingerprint chunks once. This drives Phase 1 walltime scaling
        # and the Phase 5 inference-array width.
        n_chunks = self._count_fp_chunks()
        throttle = int(self.cfg["scheduler"].get("array_throttle", 20))
        if n_chunks < 1:
            if self.dry_run:
                print("  [warn] could not count fingerprint chunks; "
                      "using 1 for this dry-run.")
                n_chunks = 1
            else:
                raise RuntimeError(
                    "Could not count fingerprint chunk files (*.txt) in "
                    f"library.fingerprint_dir="
                    f"{os.path.expandvars(self.cfg['library']['fingerprint_dir'])!r}. "
                    "Phase 5 needs this to size the inference job array. Check "
                    "the path exists and holds the prepared fingerprint files."
                )
        print(f"  Library fingerprint chunks: {n_chunks} "
              f"(Phase 5 inference array width)\n")
        if n_chunks > 1000:
            print(f"  [note] {n_chunks} inference-array tasks. SLURM caps the "
                  f"array index (MaxArraySize, often 1001). If submission is "
                  f"rejected, prepare the library into fewer/larger fingerprint "
                  f"chunk files, or ask the admins to raise MaxArraySize.\n")

        # Phases 1 and 2 are single jobs. Phases 3, 4 and 5 each expand into a
        # generate/array/collect trio and have their own submitters:
        #   3 -> 3a (shard)    + 3b (docking array)   + 3c (merge)
        #   4 -> 4a (labels)   + 4b (training array)  + 4c (evaluate)
        #   5 -> 5a (pred-gen) + 5b (inference array)

        for iteration in range(start_iter, self.total_iter + 1):
            print(f"-- Iteration {iteration} ------------------------------")
            phase_start = start_phase if iteration == start_iter else 1

            for phase_num in (1, 2):
                if phase_num < phase_start:
                    continue
                print(f"  Phase {phase_num}: {PHASES[phase_num]}")
                if phase_num == 1:
                    # Phase 1 walltime scales with the library chunk count.
                    script = self.factory.phase1_sampling(iteration, n_chunks)
                else:
                    script = self.factory.phase2_ligand_prep(iteration)
                last_job_id = self._submit_phase(
                    iteration, phase_num, script, last_job_id
                )

            # Phase 3: shard + docking array + merge.
            if phase_start <= 3:
                last_job_id = self._submit_phase3(
                    iteration, last_job_id, throttle, resume=resume_mode
                )

            # Phase 4: labels + training array + evaluation.
            if phase_start <= 4:
                last_job_id = self._submit_phase4(
                    iteration, last_job_id, throttle, resume=resume_mode
                )

            # Phase 5: generate inference scripts + run them as a job array.
            if phase_start <= 5:
                last_job_id = self._submit_phase5(
                    iteration, last_job_id, n_chunks, throttle, resume=resume_mode
                )
            print()

        # Final extraction, depends on the last iteration's phase 5.
        print("-- Final extraction -------------------------------")
        if resume_mode and self.state.data.get("final_extraction_job_id"):
            final_id = self.state.data["final_extraction_job_id"]
            print(f"  [done] final extraction already COMPLETED "
                  f"(job {final_id}) - keeping.")
        else:
            final_script = self.factory.final_extraction(self.total_iter)
            final_path   = self._write_script("final_extraction", final_script)
            final_id     = self.scheduler.submit(final_path, last_job_id)
            if not self.dry_run:
                self.state.data["final_extraction_job_id"] = final_id
                self.state.save()

        print(f"\n{'='*60}")
        print(f"  All jobs submitted. Final job ID: {final_id}")
        print(f"  State log: {self.state.path}")
        print(f"  Job scripts: {self.scripts_dir}")
        print(f"{'='*60}\n")

    def _find_resume_job_id(self, start_iter: int,
                            start_phase: int) -> str | None:
        """
        Walk backwards from (start_iter, start_phase - 1) through the state
        log to find the most recent successfully submitted job ID.
        That ID becomes the dependency for the first newly submitted phase.
        """
        # FIX: the original used a confusing nested while loop.
        # A single flat iteration over (iteration, phase) pairs in reverse
        # is easier to follow and does exactly the same thing.
        for it in range(start_iter, 0, -1):
            if it == start_iter:
                # Only phases strictly before start_phase in this iteration.
                cands = list(range(start_phase - 1, 0, -1))
            else:
                # A fully-submitted earlier iteration ends with the Phase 5b
                # inference array (5b <- 5a <- 4c <- 4b <- 4a <- phase 3 ...).
                cands = ["5b", "5a", "4c", "4b", "4a", 3, 2, 1]
            for ph in cands:
                jid = self.state.get_job_id(it, ph)
                if jid:
                    return jid
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep Docking active-learning campaign orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run a full campaign:
              python dd_orchestrator.py --config campaign.yaml

              # Resume from iteration 3, phase 4 (after a crash in training):
              python dd_orchestrator.py --config campaign.yaml --start-iter 3 --start-phase 4

              # Preview all job scripts without submitting:
              python dd_orchestrator.py --config campaign.yaml --dry-run
        """)
    )
    parser.add_argument("--config",      required=True,
                        help="Path to campaign YAML config file")
    parser.add_argument("--start-iter",  type=int, default=1,
                        help="Iteration to start from (default: 1)")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Phase within start-iter to start from (default: 1)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Write job scripts but do not submit them")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume after cancellation: query the scheduler for "
                             "each phase's real state, keep the COMPLETED ones, "
                             "and resubmit from the first incomplete phase with a "
                             "fresh dependency chain. Overrides --start-iter/-phase.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    orchestrator = DDOrchestrator(cfg, dry_run=args.dry_run)
    orchestrator.run(start_iter=args.start_iter, start_phase=args.start_phase,
                     resume=args.resume)


if __name__ == "__main__":
    main()