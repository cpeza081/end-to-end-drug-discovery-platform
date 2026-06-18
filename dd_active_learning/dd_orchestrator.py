#!/usr/bin/env python3
"""
dd_orchestrator.py
==================
Deep Docking active-learning campaign orchestrator.

Reads a YAML config file and drives the DD loop:
  Iteration 1:  Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  Iteration N:  Phase 1 (from previous predictions) → Phase 2 → 3 → 4 → 5
  Final:        extract SMILES of surviving virtual hits

The orchestrator submits one job per phase, using the scheduler's native
dependency mechanism. All job IDs are logged to <project_dir>/campaign_state.json 
so a crashed run can be resumed from the last completed phase.

Scheduler support
-----------------
SLURM  — full support (afterok dependencies, sbatch)
PBS    — basic support (afterok dependencies, qsub)
SGE    — basic support (hold_jid dependencies, qsub)

Usage
-----
  # Start a new campaign (or resume a crashed one):
  python dd_orchestrator.py --config campaign.yaml

  # Start from a specific iteration / phase (useful after manual fixes):
  python dd_orchestrator.py --config campaign.yaml --start-iter 3 --start-phase 4

  # Dry run: print the job scripts without submitting:
  python dd_orchestrator.py --config campaign.yaml --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml  # pip install pyyaml


# =============================================================================
# Config loading & path expansion
# =============================================================================

def load_config(path: str) -> dict:
    """Load YAML config, expanding environment variables in string values."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand_env(raw)

# We expand environment variables (name/value pair that operating system keeps for current shell or process).
# Programs can read them to find settings like file locations, tool paths, or runtime options. In this case,
# we let the YAML config refer to values like $HOME or ${DATA_DIR} without hardcoding them. 
def _expand_env(obj):
    """Recursively expand $VAR / ${VAR} in all string values."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(i) for i in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


# =============================================================================
# Campaign state  (persisted to JSON so we can resume)
# =============================================================================

class CampaignState:
    """
    Tracks which phases have been submitted / completed and stores job IDs.
    Written to <project_dir>/campaign_state.json after every submission.
    """

    def __init__(self, project_dir: str):
        self.path = Path(project_dir) / "campaign_state.json"
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"iterations": {}, "submitted_at": str(datetime.now())}

    # We save the state after every job submission.
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    # When we submit a job, we record its ID and timestamp as well as iteration and phase.
    def record_job(self, iteration: int, phase: int, job_id: str):
        key = str(iteration)
        self.data["iterations"].setdefault(key, {})
        self.data["iterations"][key][f"phase{phase}_job_id"] = job_id
        self.data["iterations"][key][f"phase{phase}_submitted"] = str(datetime.now())
        self.save()

    # When resuming, we can look up the last submitted job ID for a given iteration and phase to chain the next job from. 
    def get_job_id(self, iteration: int, phase: int) -> str | None:
        return (self.data["iterations"]
                .get(str(iteration), {})
                .get(f"phase{phase}_job_id"))

    # This allows us to check if a phase has already been submitted, so we don't accidentally submit duplicate jobs when resuming. 
    def is_phase_submitted(self, iteration: int, phase: int) -> bool:
        return self.get_job_id(iteration, phase) is not None


# =============================================================================
# Scheduler abstraction
# =============================================================================

class Scheduler:
    """
    Thin wrapper around SLURM / PBS / SGE submission commands.
    All scheduler-specific syntax is isolated here — the rest of the code
    is scheduler-agnostic.
    """

    def __init__(self, stype: str, account: str, dry_run: bool = False):
        self.stype = stype.upper() # Convert scheduler type to uppercase for consistency (e.g., "slurm" → "SLURM")
        self.account = account
        self.dry_run = dry_run
        if self.stype not in ("SLURM", "PBS", "SGE"):
            raise ValueError(f"Unsupported scheduler: {stype}")

    # ------------------------------------------------------------------
    # Submit a script, optionally depending on a previous job ID.
    # Returns the new job ID string.
    # ------------------------------------------------------------------
    def submit(self, script_path: str, depends_on: str | None = None) -> str:
        cmd = self._build_submit_cmd(script_path, depends_on) # Build the appropriate submission command based on the scheduler type and dependency. 
        print(f"  Submitting: {' '.join(cmd)}")

        if self.dry_run:
            fake_id = f"DRY_{Path(script_path).stem}"
            print(f"  [dry-run] Would submit → fake job ID: {fake_id}")
            return fake_id

        result = subprocess.run(cmd, capture_output=True, text=True, check=True) # Execute the submission command and capture the output, which contains the job ID assigned by the scheduler. 
        job_id = self._parse_job_id(result.stdout.strip())
        print(f"  → Job ID: {job_id}")
        return job_id

    def _build_submit_cmd(self, script: str, depends_on: str | None) -> list[str]:
        """Construct the appropriate submission command based on the scheduler type and dependency."""

        # Builds command used to submit a job script to the scheduler. 
        if self.stype == "SLURM":
            cmd = ["sbatch"] # starts the command with the submit tool for SLURM.
            if depends_on: # Checks whether this job should wait for another job first (i.e., if depends_on is not None).
                cmd += [f"--dependency=afterok:{depends_on}"] # Adds a dependency option so this job only runs after the named job succeeds.
            cmd.append(script) # Adds the script file path to the command. 

        elif self.stype == "PBS":
            cmd = ["qsub"]
            if depends_on:
                cmd += [f"-W", f"depend=afterok:{depends_on}"]
            cmd.append(script)

        elif self.stype == "SGE":
            cmd = ["qsub"]
            if depends_on:
                cmd += ["-hold_jid", depends_on]
            cmd.append(script)

        return cmd

    def _parse_job_id(self, stdout: str) -> str:
        """Extract numeric job ID from submission output."""
        if self.stype == "SLURM":
            # "Submitted batch job 12345"
            return stdout.split()[-1]
        elif self.stype == "PBS":
            # "12345.cluster"
            return stdout.split(".")[0]
        elif self.stype == "SGE":
            # "Your job 12345 (\"name\") has been submitted"
            return stdout.split()[2]
        return stdout

    # ------------------------------------------------------------------
    # Generate the scheduler header block for a job script
    # ------------------------------------------------------------------
    def header(self, job_name: str, walltime: str, nodes: int,
               cpus: int, mem: str, gpus: int, account: str,
               partition: str, log_dir: str) -> str:

        if self.stype == "SLURM":
            gpu_line = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else ""
            return textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --account={account}
                #SBATCH --partition={partition}
                #SBATCH --nodes={nodes}
                #SBATCH --cpus-per-task={cpus}
                #SBATCH --mem={mem}
                #SBATCH --time={walltime}
                #SBATCH --output={log_dir}/{job_name}_%j.out
                #SBATCH --error={log_dir}/{job_name}_%j.err
                {gpu_line}
            """).rstrip()

        elif self.stype == "PBS":
            gpu_line = f"#PBS -l ngpus={gpus}" if gpus > 0 else ""
            return textwrap.dedent(f"""\
                #!/bin/bash
                #PBS -N {job_name}
                #PBS -A {account}
                #PBS -q {partition}
                #PBS -l nodes={nodes}:ppn={cpus}
                #PBS -l mem={mem}
                #PBS -l walltime={walltime}
                #PBS -o {log_dir}/{job_name}.out
                #PBS -e {log_dir}/{job_name}.err
                {gpu_line}
            """).rstrip()

        elif self.stype == "SGE":
            gpu_line = f"#$ -l gpu={gpus}" if gpus > 0 else ""
            return textwrap.dedent(f"""\
                #!/bin/bash
                #$ -N {job_name}
                #$ -A {account}
                #$ -q {partition}
                #$ -pe smp {cpus}
                #$ -l h_vmem={mem}
                #$ -l h_rt={walltime}
                #$ -o {log_dir}/{job_name}.out
                #$ -e {log_dir}/{job_name}.err
                {gpu_line}
            """).rstrip()


# =============================================================================
# Job script generators  (one per phase)
# =============================================================================

class JobScriptFactory:
    """
    Generates the body of each phase's job script.
    All DD command calls mirror the Gentile et al. 2022 protocol exactly,
    with paths and parameters substituted from config.
    """

    def __init__(self, cfg: dict, scheduler: Scheduler):
        self.cfg = cfg
        self.s = scheduler

        # Frequently referenced config sub-trees
        self.dd = cfg["dd"]
        self.env = cfg["env"]
        self.dock = cfg["docking"]
        self.proj = cfg["project_dir"]
        self.lib = cfg["library"]
        self.res = cfg["scheduler"]["resources"]
        self.wt = cfg["scheduler"]["walltime"]
        self.sched = cfg["scheduler"]
        self.name = cfg["campaign_name"]

    # ------------------------------------------------------------------
    # Shared preamble written at the top of every script
    # ------------------------------------------------------------------
    def _preamble(self, iteration: int) -> str:
        oe_dir = self.env["openeye_dir"]
        conda_env = self.env["conda_env"]
        dd_dir = self.env["dd_protocol_dir"]
        project_dir = self.proj
        return textwrap.dedent(f"""\

            # ── Environment setup ──────────────────────────────────────────
            export DD_PROJECT_DIR="{project_dir}"
            export DD_ITERATION={iteration}
            export DD_CAMPAIGN="{self.name}"
            export PATH="{oe_dir}:$PATH"
            export OE_LICENSE="{oe_dir}/oe_license.txt"
            export DD_PROTOCOL_DIR="{dd_dir}"

            # Activate conda environment
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "{conda_env}"

            # Abort on any error
            set -euo pipefail

            echo "[$(date)] Starting iteration ${{DD_ITERATION}}"
        """)

    # ------------------------------------------------------------------
    # Phase 1: Random sampling from library (iter 1) or predictions (iter N>1)
    # ------------------------------------------------------------------
    def phase1_sampling(self, iteration: int) -> str:
        r = self.res["phase1_sampling"]
        partition = self.sched["cpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_i{iteration:02d}_p1_sampling"

        # In iteration 1 we sample from the full fingerprint library.
        # In subsequent iterations we sample from the previous iteration's
        # predicted virtual hits (morgan_1024_predictions folder).
        if iteration == 1:
            data_dir = self.lib["fingerprint_dir"]
            tot_sampling = self.dd["train_size"] + 2 * self.dd["val_size"]
        else:
            prev = iteration - 1
            data_dir = (f"{self.proj}/iteration_{prev:02d}"
                        f"/morgan_1024_predictions")
            # After iteration 1, only augment training; val/test stay fixed
            tot_sampling = self.dd["train_size"]

        proj_dir = self.proj
        proj_name = self.name
        dd_dir = self.env["dd_protocol_dir"]
        ncpu = r["cpus"]
        train_sz = self.dd["train_size"]
        val_sz = self.dd["val_size"]
        fp_dir = self.lib["fingerprint_dir"]
        smiles_dir = self.lib["smiles_dir"]

        header = self.s.header(
            job_name, self.wt["phase1_sampling"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Phase 1: Sampling (iteration {iteration}) ──────────────────
            # Determine how many molecules to sample from each library chunk,
            # then perform the actual random sampling, deduplicate, and extract
            # both Morgan fingerprints and SMILES for the sampled molecules.

            ITER_DIR="{proj_dir}/iteration_{iteration:02d}"
            mkdir -p "$ITER_DIR"

            # Step 1a: count molecules per file to reach target sample size
            python "{dd_dir}/scripts_1/molecular_file_count_updated.py" \\
                --project_name "{proj_name}" \\
                --n_iteration {iteration} \\
                --data_directory "{data_dir}" \\
                --tot_process {ncpu} \\
                --tot_sampling {tot_sampling}

            # Step 1b: perform the random sampling
            python "{dd_dir}/scripts_1/sampling.py" \\
                --project_name "{proj_name}" \\
                --file_path "{proj_dir}" \\
                --n_iteration {iteration} \\
                --data_directory "{data_dir}" \\
                --tot_process {ncpu} \\
                --train_size {train_sz} \\
                --val_size {val_sz}

            # Step 1c: remove overlaps between train / val / test sets
            python "{dd_dir}/scripts_1/sanity_check.py" \\
                --project_name "{proj_name}" \\
                --file_path "{proj_dir}" \\
                --n_iteration {iteration}

            # Step 1d: extract Morgan fingerprints for sampled molecules
            python "{dd_dir}/scripts_1/extracting_morgan.py" \\
                --project_name "{proj_name}" \\
                --file_path "{proj_dir}" \\
                --n_iteration {iteration} \\
                --morgan_directory "{fp_dir}" \\
                --tot_process {ncpu}

            # Step 1e: extract SMILES for sampled molecules
            python "{dd_dir}/scripts_1/extracting_smiles.py" \\
                --project_name "{proj_name}" \\
                --file_path "{proj_dir}" \\
                --n_iteration {iteration} \\
                --smile_directory "{smiles_dir}" \\
                --tot_process {ncpu}

            echo "[$(date)] Phase 1 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 2: 3D conformer generation with OMEGA
    # ------------------------------------------------------------------
    def phase2_ligand_prep(self, iteration: int) -> str:
        r = self.res["phase2_ligand_prep"]
        partition = self.sched["cpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_i{iteration:02d}_p2_ligprep"
        program = self.dock["program"].upper()
        proj_dir = self.proj
        ncpu = r["cpus"]

        # Ligand prep output format differs by docking program
        if program == "FRED":
            # OMEGA pose mode → oeb.gz  (pose mode generates multiple conformers
            # pre-filtered for receptor shape — best for FRED)
            omega_cmd = textwrap.dedent(f"""\
                # Generate 3D conformers in OMEGA pose mode (for FRED docking)
                for SMI_FILE in "$ITER_DIR/smile/"*.smi; do
                    BASE=$(basename "$SMI_FILE" .smi)
                    oeomega pose \\
                        -in  "$SMI_FILE" \\
                        -out "$ITER_DIR/sdf/${{BASE}}.oeb.gz" \\
                        -strictstereo false \\
                        -mpi_np {ncpu}
                done
            """)
        elif program == "GLIDE":
            # OMEGA classic mode → sdf  (one conformer per molecule)
            omega_cmd = textwrap.dedent(f"""\
                # Generate 3D conformers in OMEGA classic mode (for GLIDE docking)
                for SMI_FILE in "$ITER_DIR/smile/"*.smi; do
                    BASE=$(basename "$SMI_FILE" .smi)
                    oeomega classic \\
                        -in  "$SMI_FILE" \\
                        -out "$ITER_DIR/sdf/${{BASE}}.sdf" \\
                        -maxconfs 1 \\
                        -strictstereo false \\
                        -mpi_np {ncpu}
                done
            """)
        else:
            raise ValueError(f"Unknown docking program: {program}")

        header = self.s.header(
            job_name, self.wt["phase2_ligand_prep"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Phase 2: Ligand preparation — OMEGA conformers (iteration {iteration}) ──
            # OMEGA enumerates low-energy 3D conformations from the 2D SMILES.
            # These conformers are required as input to the docking program.

            ITER_DIR="{proj_dir}/iteration_{iteration:02d}"
            mkdir -p "$ITER_DIR/sdf"

        """) + omega_cmd + textwrap.dedent(f"""\

            echo "[$(date)] Phase 2 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 3: Docking
    # ------------------------------------------------------------------
    def phase3_docking(self, iteration: int) -> str:
        r = self.res["phase3_docking"]
        partition = self.sched["cpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_i{iteration:02d}_p3_docking"
        program = self.dock["program"].upper()
        grid = self.dock["grid_file"]
        proj_dir = self.proj
        ncpu = r["cpus"]

        if program == "FRED":
            docking_cmd = textwrap.dedent(f"""\
                # Dock each conformer file produced by Phase 2
                for OEB_FILE in "$ITER_DIR/sdf/"*.oeb.gz; do
                    BASE=$(basename "$OEB_FILE" .oeb.gz)
                    fred \\
                        -receptor "{grid}" \\
                        -dbase "$OEB_FILE" \\
                        -docked_molecule_file "$ITER_DIR/docked/${{BASE}}_docked.sdf" \\
                        -hitlist_size 0 \\
                        -mpi_np {ncpu}
                done
            """)
        elif program == "GLIDE":
            dd_dir = self.env["dd_protocol_dir"]
            glide_tmpl = self.dock.get("glide_template", "")
            docking_cmd = textwrap.dedent(f"""\
                # Generate GLIDE input scripts, then dock
                python "{dd_dir}/scripts_1/input_glide.py" \\
                    --project_name "{self.name}" \\
                    --file_path "{proj_dir}" \\
                    --grid_file "{grid}" \\
                    --iteration_no {iteration} \\
                    --glide_input "{glide_tmpl}"

                cd "$ITER_DIR/docked"
                for GLIDE_IN in *.in; do
                    "$SCHRODINGER/glide" -OVERWRITE -JOBNAME "${{GLIDE_IN%.in}}" "$GLIDE_IN"
                done
            """)
        else:
            raise ValueError(f"Unknown docking program: {program}")

        header = self.s.header(
            job_name, self.wt["phase3_docking"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Phase 3: Molecular docking (iteration {iteration}) ──────────
            # Docks the sampled molecules (training + val + test in iter 1,
            # training augmentation only in later iterations).
            # Outputs one SDF file per input set inside the "docked" folder.
            # The SDF must contain the docking score field used in Phase 4.

            ITER_DIR="{proj_dir}/iteration_{iteration:02d}"
            mkdir -p "$ITER_DIR/docked"

        """) + docking_cmd + textwrap.dedent(f"""\

            echo "[$(date)] Phase 3 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 4: DNN model training
    # ------------------------------------------------------------------
    def phase4_training(self, iteration: int) -> str:
        r = self.res["phase4_training"]
        partition = self.sched["gpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_i{iteration:02d}_p4_training"

        dd_dir = self.env["dd_protocol_dir"]
        proj_dir = self.proj
        fp_dir = self.lib["fingerprint_dir"]
        score_kw = self.dock["score_keyword"]
        total_iter = self.dd["total_iterations"]
        num_models = self.dd["num_models"]
        val_sz = self.dd["val_size"]
        pct_first = self.dd["percent_first"]
        pct_last = self.dd["percent_last"]
        recall = self.dd["recall"]
        is_last = str(iteration == total_iter).capitalize()  # "True" / "False"

        # How many docking SDF files exist — one per molecular set
        # (train + val + test = 3 in iter 1; just train in later iters)
        n_docking_files = 3 if iteration == 1 else 1

        header = self.s.header(
            job_name, self.wt["phase4_training"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Phase 4: DNN model training (iteration {iteration}) ─────────
            # 4a: Extract binary labels (virtual hit / non-hit) from SDF scores.
            #     The score_keyword must match the SDF field name exactly.
            # 4b: Train {num_models} DNN models with different hyperparameters
            #     via grid search, then select the best-performing model.
            #
            # The DNN learns to predict docking scores from Morgan fingerprints,
            # enabling fast inference over the full library in Phase 5.

            ITER_DIR="{proj_dir}/iteration_{iteration:02d}"

            # Step 4a: convert SDF docking scores → binary label files
            python "{dd_dir}/scripts_2/extract_labels.py" \\
                --project_name "{self.name}" \\
                --file_path "{proj_dir}" \\
                --iteration_no {iteration} \\
                --tot_process {n_docking_files} \\
                --score_keyword '{score_kw}'

            # Step 4b: generate training job scripts for all {num_models} models
            python "{dd_dir}/scripts_2/simple_job_models_manual.py" \\
                --iteration_no {iteration} \\
                --morgan_directory "{fp_dir}" \\
                --file_path "{proj_dir}/{self.name}" \\
                --number_of_hyp {num_models} \\
                --total_iterations {total_iter} \\
                --is_last {is_last} \\
                --number_mol {val_sz} \\
                --percent_first_mols {pct_first} \\
                --percent_last_mols {pct_last} \\
                --recall {recall}

            # Step 4c: run all model training scripts sequentially
            # (GPU resource is shared across them within this job allocation)
            for SCRIPT in "$ITER_DIR/simple_job/"*.sh; do
                bash "$SCRIPT"
            done

            # Step 4d: grid search — select the best model by test-set precision
            python "{dd_dir}/scripts_2/hyperparameter_result_evaluation.py" \\
                --n_iteration {iteration} \\
                --data_path "{proj_dir}/{self.name}" \\
                --morgan_directory "{fp_dir}" \\
                --number_mol {val_sz} \\
                --recall {recall}

            echo "[$(date)] Phase 4 complete — iteration {iteration}"
            echo "Best model stats:"
            cat "$ITER_DIR/best_model_stats.txt" 2>/dev/null || true
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 5: Inference over the full library
    # ------------------------------------------------------------------
    def phase5_inference(self, iteration: int) -> str:
        r = self.res["phase5_inference"]
        partition = self.sched["gpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_i{iteration:02d}_p5_inference"

        dd_dir = self.env["dd_protocol_dir"]
        proj_dir = self.proj
        fp_dir = self.lib["fingerprint_dir"]
        recall = self.dd["recall"]

        header = self.s.header(
            job_name, self.wt["phase5_inference"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Phase 5: Library-wide inference (iteration {iteration}) ─────
            # The best DNN model from Phase 4 scores every molecule in the
            # full fingerprint library.  Molecules whose predicted probability
            # of being a virtual hit falls below the recall-calibrated threshold
            # are discarded.  The surviving molecule IDs are written to
            # morgan_1024_predictions/ — this becomes the sampling pool for
            # the next iteration's Phase 1.

            ITER_DIR="{proj_dir}/iteration_{iteration:02d}"

            # Step 5a: generate one inference script per fingerprint chunk
            python "{dd_dir}/scripts_2/simple_job_predictions_manual.py" \\
                --project_name "{self.name}" \\
                --file_path "{proj_dir}" \\
                --n_iteration {iteration} \\
                --morgan_directory "{fp_dir}"

            # Step 5b: run inference on every chunk
            for SCRIPT in "$ITER_DIR/simple_job_predictions/"*.sh; do
                bash "$SCRIPT"
            done

            # Step 5c: report the number of surviving virtual hits
            N_HITS=$(ls "$ITER_DIR/morgan_1024_predictions/" | wc -l)
            echo "[$(date)] Phase 5 complete — iteration {iteration}"
            echo "Prediction files in morgan_1024_predictions: $N_HITS"
            echo "Estimated remaining molecules (from best_model_stats.txt):"
            grep "Total Left" "$ITER_DIR/best_model_stats.txt" 2>/dev/null || true
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Final phase: extract SMILES of surviving virtual hits for final docking
    # ------------------------------------------------------------------
    def final_extraction(self, last_iteration: int) -> str:
        r = self.res["final_extraction"]
        partition = self.sched["cpu_partition"]
        account = self.sched["account"]
        log_dir = f"{self.proj}/logs"
        job_name = f"{self.name}_final_extraction"

        dd_dir = self.env["dd_protocol_dir"]
        proj_dir = self.proj
        smiles_dir = self.lib["smiles_dir"]
        ncpu = r["cpus"]

        header = self.s.header(
            job_name, self.wt["final_extraction"],
            r["nodes"], r["cpus"], r["mem"], r["gpus"],
            account, partition, log_dir
        )

        body = textwrap.dedent(f"""\

            # ── Final extraction: retrieve SMILES for all surviving virtual hits ──
            # After the last DD iteration, the morgan_1024_predictions folder
            # contains IDs of molecules the DNN predicts are top-scorers.
            # This step maps those IDs back to SMILES so they can be prepared
            # for final explicit docking.

            LAST_PRED="{proj_dir}/iteration_{last_iteration:02d}/morgan_1024_predictions"

            python "{dd_dir}/utilities/final_extraction.py" \\
                -smile_dir "{smiles_dir}" \\
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

class DDOrchestrator:
    """
    Builds and submits the full DD active-learning chain.
    Each phase is chained to the previous one via scheduler dependencies,
    so no polling or cron is needed.
    """

    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg = cfg
        self.proj = cfg["project_dir"]
        self.total_iter = cfg["dd"]["total_iterations"]
        sched_cfg = cfg["scheduler"]
        self.scheduler = Scheduler(sched_cfg["type"], sched_cfg["account"], dry_run)
        self.factory = JobScriptFactory(cfg, self.scheduler)
        self.state = CampaignState(self.proj)
        self.scripts_dir = Path(self.proj) / "job_scripts"
        self.log_dir = Path(self.proj) / "logs"
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

    def _submit_phase(self, iteration: int, phase: int,
                      script_content: str, depends_on: str | None) -> str:
        """Write the script, record it, and submit it."""
        script_name = f"iter_{iteration:02d}_phase{phase}"
        path = self._write_script(script_name, script_content)

        if self.state.is_phase_submitted(iteration, phase):
            existing = self.state.get_job_id(iteration, phase)
            print(f"  [skip] iter {iteration} phase {phase} already submitted "
                  f"(job {existing}) — using existing ID for dependency chain")
            return existing

        job_id = self.scheduler.submit(path, depends_on)
        self.state.record_job(iteration, phase, job_id)
        return job_id

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, start_iter: int = 1, start_phase: int = 1):
        """
        Submit the full DD campaign.
        Each iteration submits phases 1–5 in a dependency chain.
        The final extraction is submitted after the last iteration's phase 5.
        """
        print(f"\n{'='*60}")
        print(f"  Deep Docking Campaign: {self.cfg['campaign_name']}")
        print(f"  Total iterations: {self.total_iter}")
        print(f"  Scheduler: {self.cfg['scheduler']['type']}")
        print(f"  Project dir: {self.proj}")
        print(f"{'='*60}\n")

        last_job_id = None  # job ID from previous phase / iteration

        # If resuming mid-campaign, find the last known job ID
        if start_iter > 1 or start_phase > 1:
            last_job_id = self._find_resume_job_id(start_iter, start_phase)
            print(f"Resuming from iteration {start_iter}, phase {start_phase}")
            print(f"Chaining from job ID: {last_job_id}\n")

        for iteration in range(start_iter, self.total_iter + 1):
            print(f"── Iteration {iteration} ─────────────────────────────")

            phase_start = start_phase if iteration == start_iter else 1

            phases = {
                1: self.factory.phase1_sampling,
                2: self.factory.phase2_ligand_prep,
                3: self.factory.phase3_docking,
                4: self.factory.phase4_training,
                5: self.factory.phase5_inference,
            }

            for phase_num, phase_fn in phases.items():
                if phase_num < phase_start:
                    continue  # skip phases already done when resuming

                label = {1: "Sampling", 2: "Ligand prep", 3: "Docking",
                         4: "Training", 5: "Inference"}[phase_num]
                print(f"  Phase {phase_num}: {label}")

                script = phase_fn(iteration)
                last_job_id = self._submit_phase(
                    iteration, phase_num, script, last_job_id
                )

            print()

        # Final extraction — depends on the last iteration's phase 5
        print("── Final extraction ─────────────────────────────────")
        final_script = self.factory.final_extraction(self.total_iter)
        final_path = self._write_script("final_extraction", final_script)
        final_id = self.scheduler.submit(final_path, last_job_id)
        self.state.data["final_extraction_job_id"] = final_id
        self.state.save()

        print(f"\n{'='*60}")
        print(f"  All jobs submitted. Final job ID: {final_id}")
        print(f"  State log: {self.state.path}")
        print(f"  Job scripts: {self.scripts_dir}")
        print(f"{'='*60}\n")

    def _find_resume_job_id(self, start_iter: int,
                            start_phase: int) -> str | None:
        """Find the most recent completed job ID to chain the next phase from."""
        # Walk backwards from (start_iter, start_phase - 1) to find a job ID
        phase = start_phase - 1
        iteration = start_iter
        while iteration >= 1:
            while phase >= 1:
                jid = self.state.get_job_id(iteration, phase)
                if jid:
                    return jid
                phase -= 1
            iteration -= 1
            phase = 5  # 5 phases per iteration
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
    parser.add_argument("--config", required=True,
                        help="Path to campaign YAML config file")
    parser.add_argument("--start-iter", type=int, default=1,
                        help="Iteration to start from (default: 1)")
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Phase within start-iter to start from (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write job scripts but do not submit them")
    args = parser.parse_args()

    cfg = load_config(args.config)
    orchestrator = DDOrchestrator(cfg, dry_run=args.dry_run)
    orchestrator.run(start_iter=args.start_iter, start_phase=args.start_phase)


if __name__ == "__main__":
    main()
