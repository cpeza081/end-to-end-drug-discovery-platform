#!/usr/bin/env python3
"""
dd_orchestrator.py
==================
Deep Docking active-learning campaign orchestrator.

Reads a YAML config file and drives the full DD loop:
  Iteration 1:  Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  Iteration N:  Phase 1 (from previous predictions) → Phase 2 → 3 → 4 → 5
  Final:        extract SMILES of surviving virtual hits

The orchestrator submits one job per phase, using the scheduler's native
dependency mechanism so phases run in sequence without polling.  All job IDs
are logged to <project_dir>/campaign_state.json so a crashed run can be
resumed from the last completed phase.

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
    Written to <project_dir>/campaign_state.json after every submission,
    so a crashed run can be resumed without re-submitting completed phases.
    """

    def __init__(self, project_dir: str):
        self.path = Path(project_dir) / "campaign_state.json"
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"iterations": {}, "submitted_at": str(datetime.now())}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_job(self, iteration: int, phase: int, job_id: str):
        key = str(iteration)
        self.data["iterations"].setdefault(key, {})
        self.data["iterations"][key][f"phase{phase}_job_id"] = job_id
        self.data["iterations"][key][f"phase{phase}_submitted"] = str(datetime.now())
        self.save()

    def get_job_id(self, iteration: int, phase: int) -> str | None:
        return (self.data["iterations"]
                .get(str(iteration), {})
                .get(f"phase{phase}_job_id"))

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
        self.stype = stype.upper()
        self.account = account
        self.dry_run = dry_run
        if self.stype not in ("SLURM", "PBS", "SGE"):
            raise ValueError(f"Unsupported scheduler: {stype}. "
                             f"Choose from: SLURM, PBS, SGE")

    def submit(self, script_path: str, depends_on: str | None = None) -> str:
        """Submit a script, optionally depending on a previous job ID.
        Returns the new job ID string."""
        cmd = self._build_submit_cmd(script_path, depends_on)
        print(f"  Submitting: {' '.join(cmd)}")

        if self.dry_run:
            fake_id = f"DRY_{Path(script_path).stem}"
            print(f"  [dry-run] Would submit → fake job ID: {fake_id}")
            return fake_id

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = self._parse_job_id(result.stdout.strip())
        print(f"  → Job ID: {job_id}")
        return job_id

    def _build_submit_cmd(self, script: str, depends_on: str | None) -> list[str]:
        # FIX: previously the SGE branch had no explicit return, relying on
        # fall-through to a bare `return cmd` that didn't exist.  Each branch
        # now returns explicitly, making the control flow unambiguous.
        if self.stype == "SLURM":
            cmd = ["sbatch"]
            if depends_on:
                cmd += [f"--dependency=afterok:{depends_on}"]
            cmd.append(script)
            return cmd

        if self.stype == "PBS":
            cmd = ["qsub"]
            if depends_on:
                cmd += ["-W", f"depend=afterok:{depends_on}"]
            cmd.append(script)
            return cmd

        # SGE
        cmd = ["qsub"]
        if depends_on:
            cmd += ["-hold_jid", depends_on]
        cmd.append(script)
        return cmd

    def _parse_job_id(self, stdout: str) -> str:
        """Extract the numeric job ID from the scheduler's submission output."""
        if self.stype == "SLURM":
            return stdout.split()[-1]       # "Submitted batch job 12345"
        if self.stype == "PBS":
            return stdout.split(".")[0]     # "12345.cluster.name"
        # SGE: "Your job 12345 ("name") has been submitted"
        return stdout.split()[2]

    def header(self, job_name: str, walltime: str, nodes: int,
               cpus: int, mem: str, gpus: int, account: str,
               partition: str, log_dir: str) -> str:
        """Return the scheduler-specific resource header for a job script."""
        # FIX: gpu_line is omitted entirely (not just set to "") when gpus == 0,
        # so the generated script header doesn't contain a stray blank line.
        gpu_lines = {
            "SLURM": f"#SBATCH --gres=gpu:{gpus}",
            "PBS":   f"#PBS -l ngpus={gpus}",
            "SGE":   f"#$ -l gpu={gpus}",
        }
        gpu_line = (gpu_lines[self.stype] + "\n") if gpus > 0 else ""

        if self.stype == "SLURM":
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
                {gpu_line}""")

        if self.stype == "PBS":
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
                {gpu_line}""")

        # SGE
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
            {gpu_line}""")


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


class JobScriptFactory:
    """
    Generates the body of each phase's job script.
    All DD command calls mirror the Gentile et al. 2022 protocol exactly,
    with paths and parameters substituted from config.
    """

    def __init__(self, cfg: dict, scheduler: Scheduler):
        self.cfg = cfg
        self.s = scheduler

        # FIX: previously the constructor unpacked config into short aliases
        # (self.dd, self.env, …) AND every method re-bound those to local
        # variables anyway, giving two levels of indirection with no benefit.
        # Now we hold the full config and let each method reach into it directly
        # with clear, self-documenting keys.  One level, no aliases.
        self.proj   = cfg["project_dir"]
        self.name   = cfg["campaign_name"]

    # ------------------------------------------------------------------
    # Shared preamble written at the top of every script
    # ------------------------------------------------------------------
    def _preamble(self, iteration: int) -> str:
        oe_dir    = self.cfg["env"]["openeye_dir"]
        conda_env = self.cfg["env"]["conda_env"]
        dd_dir    = self.cfg["env"]["dd_protocol_dir"]

        return textwrap.dedent(f"""\

            # ── Environment setup ──────────────────────────────────────────
            export DD_PROJECT_DIR="{self.proj}"
            export DD_ITERATION={iteration}
            export DD_CAMPAIGN="{self.name}"
            export PATH="{oe_dir}:$PATH"
            export OE_LICENSE="{oe_dir}/oe_license.txt"
            export DD_PROTOCOL_DIR="{dd_dir}"

            # Activate conda environment
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "{conda_env}"

            # Abort immediately if any command fails — this ensures the
            # scheduler marks the job as FAILED rather than silently
            # continuing into a broken state, which would break the
            # dependency chain for subsequent phases.
            set -euo pipefail

            echo "[$(date)] Starting iteration ${{DD_ITERATION}}"
        """)

    def _make_header(self, phase_key: str, job_name: str,
                     partition_key: str) -> str:
        """Build the scheduler header for any phase using config lookups."""
        r   = self.cfg["scheduler"]["resources"][phase_key]
        wt  = self.cfg["scheduler"]["walltime"][phase_key]
        acc = self.cfg["scheduler"]["account"]
        par = self.cfg["scheduler"][partition_key]
        log = f"{self.proj}/logs"
        return self.s.header(job_name, wt, r["nodes"], r["cpus"],
                             r["mem"], r["gpus"], acc, par, log)

    # ------------------------------------------------------------------
    # Phase 1: Random sampling from library (iter 1) or predictions (iter N>1)
    # ------------------------------------------------------------------
    def phase1_sampling(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p1_sampling"
        header   = self._make_header("phase1_sampling", job_name, "cpu_partition")

        dd_dir   = self.cfg["env"]["dd_protocol_dir"]
        ncpu     = self.cfg["scheduler"]["resources"]["phase1_sampling"]["cpus"]
        fp_dir   = self.cfg["library"]["fingerprint_dir"]
        smi_dir  = self.cfg["library"]["smiles_dir"]
        train_sz = self.cfg["dd"]["train_size"]
        val_sz   = self.cfg["dd"]["val_size"]

        # In iteration 1 we sample from the full fingerprint library.
        # In subsequent iterations we sample only from the previous iteration's
        # virtual-hit predictions — validation and test sets are frozen after
        # iteration 1 and reused throughout (see paper §'Molecular sample size').
        if iteration == 1:
            data_dir     = fp_dir
            tot_sampling = train_sz + 2 * val_sz
        else:
            data_dir     = f"{self.proj}/iteration_{iteration - 1:02d}/morgan_1024_predictions"
            tot_sampling = train_sz   # only augment training; val/test are fixed

        body = textwrap.dedent(f"""\

            # ── Phase 1: Sampling (iteration {iteration}) ──────────────────
            # Determine how many molecules to sample from each library chunk,
            # then perform the actual random sampling, deduplicate, and extract
            # both Morgan fingerprints and SMILES for the sampled molecules.

            ITER_DIR="{self.proj}/iteration_{iteration:02d}"
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

            echo "[$(date)] Phase 1 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 2: 3D conformer generation with OMEGA
    # ------------------------------------------------------------------
    def phase2_ligand_prep(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p2_ligprep"
        header   = self._make_header("phase2_ligand_prep", job_name, "cpu_partition")
        program  = self.cfg["docking"]["program"].upper()
        ncpu     = self.cfg["scheduler"]["resources"]["phase2_ligand_prep"]["cpus"]

        # OMEGA command differs by docking program:
        #   FRED  → pose mode, outputs .oeb.gz (receptor-filtered conformers)
        #   GLIDE → classic mode, outputs .sdf  (one conformer per molecule)
        if program == "FRED":
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
            raise ValueError(f"Unknown docking program: {program}. "
                             f"Choose FRED or GLIDE.")

        body = textwrap.dedent(f"""\

            # ── Phase 2: Ligand preparation — OMEGA conformers (iteration {iteration}) ──
            # OMEGA enumerates low-energy 3D conformations from the 2D SMILES.
            # These conformers are required as input to the docking program.

            ITER_DIR="{self.proj}/iteration_{iteration:02d}"
            mkdir -p "$ITER_DIR/sdf"

        """) + omega_cmd + textwrap.dedent(f"""\

            echo "[$(date)] Phase 2 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 3: Docking
    # ------------------------------------------------------------------
    def phase3_docking(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p3_docking"
        header   = self._make_header("phase3_docking", job_name, "cpu_partition")
        program  = self.cfg["docking"]["program"].upper()
        grid     = self.cfg["docking"]["grid_file"]
        ncpu     = self.cfg["scheduler"]["resources"]["phase3_docking"]["cpus"]

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
            dd_dir      = self.cfg["env"]["dd_protocol_dir"]
            glide_tmpl  = self.cfg["docking"].get("glide_template", "")
            docking_cmd = textwrap.dedent(f"""\
                # Generate GLIDE input scripts, then dock
                python "{dd_dir}/scripts_1/input_glide.py" \\
                    --project_name "{self.name}" \\
                    --file_path "{self.proj}" \\
                    --grid_file "{grid}" \\
                    --iteration_no {iteration} \\
                    --glide_input "{glide_tmpl}"

                cd "$ITER_DIR/docked"
                for GLIDE_IN in *.in; do
                    "$SCHRODINGER/glide" -OVERWRITE -JOBNAME "${{GLIDE_IN%.in}}" "$GLIDE_IN"
                done
            """)
        else:
            raise ValueError(f"Unknown docking program: {program}. "
                             f"Choose FRED or GLIDE.")

        body = textwrap.dedent(f"""\

            # ── Phase 3: Molecular docking (iteration {iteration}) ──────────
            # Docks the sampled molecules (training + val + test in iter 1,
            # training augmentation only in later iterations).
            # Outputs one SDF file per input set inside the "docked" folder.
            # The SDF must contain the docking score field used in Phase 4.

            ITER_DIR="{self.proj}/iteration_{iteration:02d}"
            mkdir -p "$ITER_DIR/docked"

        """) + docking_cmd + textwrap.dedent(f"""\

            echo "[$(date)] Phase 3 complete — iteration {iteration}"
        """)

        return header + self._preamble(iteration) + body

    # ------------------------------------------------------------------
    # Phase 4: DNN model training
    # ------------------------------------------------------------------
    def phase4_training(self, iteration: int) -> str:
        job_name = f"{self.name}_i{iteration:02d}_p4_training"
        header   = self._make_header("phase4_training", job_name, "gpu_partition")

        dd_dir     = self.cfg["env"]["dd_protocol_dir"]
        fp_dir     = self.cfg["library"]["fingerprint_dir"]
        score_kw   = self.cfg["docking"]["score_keyword"]
        total_iter = self.cfg["dd"]["total_iterations"]
        num_models = self.cfg["dd"]["num_models"]
        val_sz     = self.cfg["dd"]["val_size"]
        pct_first  = self.cfg["dd"]["percent_first"]
        pct_last   = self.cfg["dd"]["percent_last"]
        recall     = self.cfg["dd"]["recall"]

        # is_last controls whether the final score threshold is applied.
        # Python's bool → str gives "True"/"False" which the DD script expects.
        is_last = str(iteration == total_iter)

        # Iteration 1 docks train + val + test (3 SDF files);
        # later iterations dock only the training augmentation batch (1 file).
        n_docking_files = 3 if iteration == 1 else 1

        body = textwrap.dedent(f"""\

            # ── Phase 4: DNN model training (iteration {iteration}) ─────────
            # 4a: Extract binary labels (virtual hit / non-hit) from SDF scores.
            #     The score_keyword must match the SDF field name exactly.
            # 4b: Train {num_models} DNN models with different hyperparameters
            #     via grid search, then select the best-performing model.
            #
            # The DNN learns to predict docking scores from Morgan fingerprints,
            # enabling fast inference over the full library in Phase 5.

            ITER_DIR="{self.proj}/iteration_{iteration:02d}"

            # Step 4a: convert SDF docking scores → binary label files
            python "{dd_dir}/scripts_2/extract_labels.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
                --iteration_no {iteration} \\
                --tot_process {n_docking_files} \\
                --score_keyword '{score_kw}'

            # Step 4b: generate training job scripts for all {num_models} models
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

            # Step 4c: run all model training scripts sequentially
            # (GPU resource is shared across them within this job allocation)
            for SCRIPT in "$ITER_DIR/simple_job/"*.sh; do
                bash "$SCRIPT"
            done

            # Step 4d: grid search — select the best model by test-set precision
            python "{dd_dir}/scripts_2/hyperparameter_result_evaluation.py" \\
                --n_iteration {iteration} \\
                --data_path "{self.proj}/{self.name}" \\
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
        job_name = f"{self.name}_i{iteration:02d}_p5_inference"
        header   = self._make_header("phase5_inference", job_name, "gpu_partition")

        dd_dir = self.cfg["env"]["dd_protocol_dir"]
        fp_dir = self.cfg["library"]["fingerprint_dir"]
        recall = self.cfg["dd"]["recall"]

        body = textwrap.dedent(f"""\

            # ── Phase 5: Library-wide inference (iteration {iteration}) ─────
            # The best DNN model from Phase 4 scores every molecule in the
            # full fingerprint library.  Molecules whose predicted probability
            # of being a virtual hit falls below the recall-calibrated threshold
            # are discarded.  The surviving molecule IDs are written to
            # morgan_1024_predictions/ — this becomes the sampling pool for
            # the next iteration's Phase 1.

            ITER_DIR="{self.proj}/iteration_{iteration:02d}"

            # Step 5a: generate one inference script per fingerprint chunk
            python "{dd_dir}/scripts_2/simple_job_predictions_manual.py" \\
                --project_name "{self.name}" \\
                --file_path "{self.proj}" \\
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
        job_name = f"{self.name}_final_extraction"
        header   = self._make_header("final_extraction", job_name, "cpu_partition")

        dd_dir   = self.cfg["env"]["dd_protocol_dir"]
        smi_dir  = self.cfg["library"]["smiles_dir"]
        ncpu     = self.cfg["scheduler"]["resources"]["final_extraction"]["cpus"]

        body = textwrap.dedent(f"""\

            # ── Final extraction: retrieve SMILES for all surviving virtual hits ──
            # After the last DD iteration, the morgan_1024_predictions folder
            # contains IDs of molecules the DNN predicts are top-scorers.
            # This step maps those IDs back to SMILES so they can be prepared
            # for final explicit docking.

            LAST_PRED="{self.proj}/iteration_{last_iteration:02d}/morgan_1024_predictions"

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

class DDOrchestrator:
    """
    Builds and submits the full DD active-learning chain.
    Each phase is chained to the previous one via scheduler dependencies,
    so no polling or cron is needed.
    """

    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg        = cfg
        self.proj       = cfg["project_dir"]
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

        last_job_id = None

        if start_iter > 1 or start_phase > 1:
            last_job_id = self._find_resume_job_id(start_iter, start_phase)
            print(f"Resuming from iteration {start_iter}, phase {start_phase}")
            print(f"Chaining from job ID: {last_job_id}\n")

        # FIX: PHASES is now a module-level constant, not rebuilt each call.
        phase_methods = {
            1: self.factory.phase1_sampling,
            2: self.factory.phase2_ligand_prep,
            3: self.factory.phase3_docking,
            4: self.factory.phase4_training,
            5: self.factory.phase5_inference,
        }

        for iteration in range(start_iter, self.total_iter + 1):
            print(f"── Iteration {iteration} ─────────────────────────────")
            phase_start = start_phase if iteration == start_iter else 1

            for phase_num, phase_fn in phase_methods.items():
                if phase_num < phase_start:
                    continue

                print(f"  Phase {phase_num}: {PHASES[phase_num]}")
                script = phase_fn(iteration)
                last_job_id = self._submit_phase(
                    iteration, phase_num, script, last_job_id
                )
            print()

        # Final extraction — depends on the last iteration's phase 5
        print("── Final extraction ─────────────────────────────────")
        final_script = self.factory.final_extraction(self.total_iter)
        final_path   = self._write_script("final_extraction", final_script)
        final_id     = self.scheduler.submit(final_path, last_job_id)
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
            # For the start iteration, look only at phases before start_phase.
            # For earlier iterations, all 5 phases are candidates.
            phase_ceiling = (start_phase - 1) if it == start_iter else 5
            for ph in range(phase_ceiling, 0, -1):
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    orchestrator = DDOrchestrator(cfg, dry_run=args.dry_run)
    orchestrator.run(start_iter=args.start_iter, start_phase=args.start_phase)


if __name__ == "__main__":
    main()