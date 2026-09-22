#!/usr/bin/env python3
"""
dd_final_docking_wizard.py
===========================
Interactive setup wizard for Stage VIII (final phase) of a completed Deep
Docking campaign: pick how many of the surviving molecules to dock, and
generate + submit the SLURM job(s) that do it.

Process:

  1. connect to the campaign and sanity-check it looks finished
  2. make sure final_extraction has run, and read the candidate count
  3. ask how many top-scoring molecules to dock
  4. generate the SLURM job scripts and submit them

One independent array task per batch (default 10,000 molecules), each
within one walltime budget (default: this campaign's own
scheduler.walltime.phase3_docking, i.e. 24h out of the box), so the final docking
run can use as many simultaneous GPU allocations as the cluster grants.
Each array task calls
"dd_final_docking.py dock --array-task-id $SLURM_ARRAY_TASK_ID".

Campaign detection and state tracking (CampaignState, Scheduler,
JobScriptFactory's header/preamble builders, campaign.yaml,
campaign_state.json) reuse the exact same tools and conventions as the rest
of this package, so a final-docking run looks and behaves like every other
phase in this campaign.

Usage
-----
  # Fully interactive, detects the campaign, asks how many to dock:
  python dd_final_docking_wizard.py --config campaign.yaml

  # Non-interactive (e.g. called from another script):
  python dd_final_docking_wizard.py --config campaign.yaml \\
      --top-n 100000 --yes

  # Preview the job scripts without submitting anything:
  python dd_final_docking_wizard.py --config campaign.yaml --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dd_active_learning"))

from dd_orchestrator import CampaignState, JobScriptFactory, Scheduler
from dd_status import query_job_status
from dd_utils import count_lines_fast, load_config


# ---------------------------------------------------------------------------
# Small interactive-prompt helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            # No terminal attached (e.g. invoked with stdin redirected from
            # something else, or from a non-interactive job). Fall back to
            # the default.
            if default is not None:
                print(f"\n(no input available -- using default: {default})")
                return default
            raise SystemExit(
                "\nERROR: no input available for a required prompt and no "
                "default to fall back to. Pass the equivalent CLI flag "
                "(e.g. --top-n) to run this non-interactively.")
        if not raw and default is not None:
            return default
        if not raw:
            print("Please enter a value.")
            continue
        try:
            return cast(raw)
        except ValueError:
            print(f"Could not parse '{raw}', try again.")


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        raw = input(f"{prompt}{suffix}: ").strip().lower()
    except EOFError:
        print(f"\n(no input available -- using default: "
              f"{'yes' if default else 'no'})")
        return default
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------

class FinalDockingWizard:
    """
    Mirrors DDOrchestrator's structure (same Scheduler/JobScriptFactory/
    CampaignState building blocks, same script-write-then-submit pattern)
    but drives the one-off final-docking stage instead of the iteration
    loop. The job scripts it writes are shells that call
    into dd_final_docking.py, this package's execution engine.
    """

    def __init__(self, cfg: dict, config_path: str, dry_run: bool = False):
        self.cfg = cfg
        self.config_path = config_path
        # dd_final_docking.py lives alongside this wizard.
        self.engine = Path(__file__).resolve().parent / "dd_final_docking.py"
        self.proj = Path(cfg["project_dir"])
        self.total_iter = cfg["dd"]["total_iterations"]
        self.scheduler = Scheduler(cfg["scheduler"]["type"],
                                    cfg["scheduler"]["account"], dry_run)
        self.factory = JobScriptFactory(cfg, self.scheduler)
        self.state = CampaignState(str(self.proj))
        self.final_dir = self.proj / "final_docking"
        self.scripts_dir = self.proj / "job_scripts"
        self.log_dir = self.proj / "logs"
        self.last_iteration = self.total_iter

    # ------------------------------------------------------------------
    # Step 1: connect to the campaign and sanity-check it looks finished
    # ------------------------------------------------------------------
    def _connect(self, assume_yes: bool, iteration_override: int | None = None):
        print(f"\nCampaign: {self.cfg['campaign_name']}")
        print(f"Project dir: {self.proj}")
        print(f"Configured total_iterations: {self.total_iter}")

        if iteration_override is not None:
            self.last_iteration = iteration_override
            print(f"Using --iteration {iteration_override} (explicit), "
                  "skipping auto-detection.")
            return

        # Phase 5 (inference) is itself split into 5a_generate (per-chunk
        # script generation) and 5b_array (the actual inference array job).
        # CampaignState records these under "phase5a_job_id" /
        # "phase5b_job_id".
        last_with_phase5 = None
        for it in range(self.total_iter, 0, -1):
            if self.state.get_job_id(it, "5b"):
                last_with_phase5 = it
                break

        if last_with_phase5 is None:
            print(f"No iteration has a recorded phase-5b (inference array) "
                  f"job in {self.state.path}. This doesn't look like a run "
                  "launched with dd_orchestrator.py. If you stopped the "
                  "campaign early and already ran final_extraction by "
                  "hand, pass --iteration N to skip this check.")
            if not _confirm("Continue anyway?", default=False):
                sys.exit(1)
            return

        status = query_job_status(self.state.get_job_id(last_with_phase5, "5b"),
                                   self.cfg["scheduler"]["type"])
        print(f"Iteration {last_with_phase5}/{self.total_iter}: phase 5b "
              f"(inference array) status = {status}")

        if last_with_phase5 != self.total_iter:
            print(f"WARNING: total_iterations is {self.total_iter} but "
                  f"iteration {last_with_phase5} is the latest with phase 5b "
                  "submitted -- the campaign may not be finished.")
        if status != "COMPLETED" and not assume_yes:
            if not _confirm(f"Phase 5b status is '{status}', not COMPLETED. "
                             "Continue anyway?", default=False):
                sys.exit(1)

        self.last_iteration = last_with_phase5

    # ------------------------------------------------------------------
    # Step 2: make sure final_extraction has been run, and get the count
    # ------------------------------------------------------------------
    def _ensure_final_extraction(self, assume_yes: bool):
        smiles_path = self.proj / "smiles.csv"
        id_score_path = self.proj / "id_score.csv"

        if smiles_path.exists() and id_score_path.exists():
            n = count_lines_fast(id_score_path) - 1  # minus header
            print(f"\nFound existing final_extraction output: {n:,} "
                  f"candidate molecule(s) in {id_score_path}.")
            return n

        print(f"\nNo smiles.csv/id_score.csv in {self.proj} yet -- "
              "final_extraction hasn't been run.")

        existing_job = self.state.data.get("final_extraction_job_id")
        if existing_job:
            status = query_job_status(existing_job,
                                       self.cfg["scheduler"]["type"])
            print(f"final_extraction was already submitted as job "
                  f"{existing_job} (status: {status}).")
            print("Wait for it to finish, then re-run this wizard.")
            return None

        if not assume_yes and not _confirm("Submit final_extraction now?",
                                            default=True):
            print("Not submitting. Re-run this wizard once final_extraction "
                  "has been run.")
            return None

        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        depends_on = self.state.get_job_id(self.last_iteration, "5b")
        script = self.factory.final_extraction(self.last_iteration)

        # final_extraction.py writes smiles.csv/id_score.csv into its
        # current working directory, which is the sbatch CALLER's cwd
        # unless the job itself cd's, not necessarily project_dir.
        # dd_status.py (and this wizard) both expect them in project_dir,
        # so make that explicit here.
        anchor = 'echo "[$(date)] Starting iteration ${DD_ITERATION}"'
        script = script.replace(
            anchor,
            anchor + f'\ncd "{self.proj}"  # final_extraction writes here\n')

        job_id = self._submit("final_extraction", script, depends_on)
        self.state.data["final_extraction_job_id"] = job_id
        self.state.save()
        print(f"Submitted final_extraction as job {job_id}.")
        print("Re-run this wizard once it completes to continue.")
        return None

    # ------------------------------------------------------------------
    # Step 3: ask how many of the top-scoring molecules to dock
    # ------------------------------------------------------------------
    def _choose_top_n(self, n_available: int, top_n, assume_yes: bool) -> int:
        if top_n is not None:
            if top_n > n_available:
                print(f"WARNING: requested --top-n={top_n} but only "
                      f"{n_available:,} molecules are available; using all "
                      "of them.")
                top_n = n_available
            print(f"Docking top {top_n:,} of {n_available:,} molecules "
                  "(from --top-n).")
            return top_n

        print(f"\n{n_available:,} molecules survived the final extraction.")
        while True:
            n = _ask("How many of the top-scoring molecules do you want to "
                      "dock?", default=str(n_available), cast=int)
            if 1 <= n <= n_available:
                return n
            print(f"Enter a number between 1 and {n_available:,}.")

    # ------------------------------------------------------------------
    # Step 4: slice the top-N and chunk it, by delegating to
    # dd_final_docking.py's own "select" subcommand.
    # ------------------------------------------------------------------
    def _select_and_chunk(self, top_n: int, batch_size: int) -> int:
        smile_dir = self.final_dir / "smile"
        if smile_dir.is_dir():
            for old in smile_dir.glob("chunk_*.smi"):
                old.unlink()  # clean slate, since a re-run with a different
                              # top-n must not leave stale extra chunks
                              # behind

        cmd = [sys.executable, str(self.engine), "select",
               "--smiles", str(self.proj / "smiles.csv"),
               "--id-score", str(self.proj / "id_score.csv"),
               "--top-n", str(top_n),
               "--final-dir", str(self.final_dir),
               "--batch-size", str(batch_size)]
        print("\nRunning: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        return len(sorted(smile_dir.glob("chunk_*.smi")))

    # ------------------------------------------------------------------
    # Job script builders
    # ------------------------------------------------------------------
    def _header_with_array(self, resource_key: str, job_name: str,
                            partition_key: str, walltime: str,
                            array_expr: str) -> str:
        # Delegate to JobScriptFactory._make_header
        header = self.factory._make_header(resource_key, job_name,
                                            partition_key,
                                            walltime_override=walltime,
                                            array_log=True)
        return header + f"#SBATCH --array={array_expr}\n"

    def _build_ligprep_script(self) -> str:
        job_name = f"{self.cfg['campaign_name']}_final_ligprep"
        header = self.factory._make_header("phase2_ligand_prep", job_name,
                                            "cpu_partition")
        ncpu = self.cfg["scheduler"]["resources"]["phase2_ligand_prep"]["cpus"]

        body = textwrap.dedent(f"""

            # -- Final ligand preparation --------------------------------
            # Delegates to dd_final_docking.py's own "prepare" subcommand
            # (this package's single execution engine for final docking).
            # With --campaign-config it shells out to the same tool Phase 2
            # uses every iteration (dd_ligand_prep.py: RDKit ETKDG + Meeko)
            # and picks the right output format (sdf for Gnina, pdbqt for
            # AutoDock-GPU) from campaign.yaml's docking.program.
            FINAL_DIR="{self.final_dir}"

            python "{self.engine}" prepare \\
                --final-dir "$FINAL_DIR" \\
                --campaign-config "{self.config_path}" \\
                --nprocs {ncpu}

            echo "[$(date)] Final ligand prep complete"
        """)
        return header + self.factory._preamble(self.last_iteration) + body

    def _build_docking_array_script(self, n_chunks: int, batch_size: int,
                                     walltime: str, max_concurrent) -> str:
        job_name = f"{self.cfg['campaign_name']}_final_docking"
        array_expr = f"0-{n_chunks - 1}"
        if max_concurrent:
            array_expr += f"%{max_concurrent}"
        header = self._header_with_array("phase3_docking", job_name,
                                          "gpu_partition", walltime,
                                          array_expr)

        body = textwrap.dedent(f"""

            # -- Final docking array task ---------------------------------
            # Docks exactly one chunk (up to {batch_size:,} molecules) of
            # the top-N pool this wizard selected, by delegating to
            # dd_final_docking.py's "dock" subcommand. One independent SLURM array task per
            # chunk, so a large final-docking run can use as many simultaneous GPU
            # allocations as the cluster grants.
            FINAL_DIR="{self.final_dir}"

            python "{self.engine}" dock \\
                --final-dir "$FINAL_DIR" \\
                --campaign-config "{self.config_path}" \\
                --array-task-id "$SLURM_ARRAY_TASK_ID"

            echo "[$(date)] Final docking task $SLURM_ARRAY_TASK_ID complete"
        """)
        # gpu=True adds the same nvidia-smi health check every other
        # GPU-partition phase gets (phase3b_array, phase4b_array, ...)
        return (header + self.factory._preamble(self.last_iteration, gpu=True)
                + body)

    def _build_merge_script(self, top_n: int) -> str:
        job_name = f"{self.cfg['campaign_name']}_final_merge"
        header = self.factory._make_header("final_extraction", job_name,
                                            "cpu_partition")

        body = textwrap.dedent(f"""

            # -- Final docking merge --------------------------------------
            # Delegates to dd_final_docking.py's "merge" subcommand. Both
            # docking backends have already been normalized to one scored
            # SDF per chunk by this point. Merge collapses every chunk into one
            # best-pose-per-molecule, score-sorted handoff file, ranked by
            # --campaign-config's docking.score_keyword.
            FINAL_DIR="{self.final_dir}"

            python "{self.engine}" merge \\
                --final-dir "$FINAL_DIR" \\
                --campaign-config "{self.config_path}" \\
                --out "$FINAL_DIR/final_top{top_n}_docked.sdf" \\
                --summary-csv "$FINAL_DIR/final_top{top_n}_scores.csv"

            echo "[$(date)] Final docking merge complete"
            echo "Handoff file: $FINAL_DIR/final_top{top_n}_docked.sdf"
        """)
        return header + self.factory._preamble(self.last_iteration) + body

    def _submit(self, name: str, script_content: str, depends_on) -> str:
        path = self.scripts_dir / f"{name}.sh"
        path.write_text(script_content)
        path.chmod(0o755)
        return self.scheduler.submit(str(path), depends_on)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, top_n, batch_size: int, walltime, max_concurrent,
            assume_yes: bool, do_merge: bool, iteration: int | None = None):
        self._connect(assume_yes, iteration_override=iteration)
        n_available = self._ensure_final_extraction(assume_yes)
        if n_available is None:
            return

        top_n = self._choose_top_n(n_available, top_n, assume_yes)
        n_chunks = self._select_and_chunk(top_n, batch_size)
        walltime = walltime or self.cfg["scheduler"]["walltime"].get(
            "phase3_docking", "24:00:00")

        print(f"\n{top_n:,} molecule(s) -> {n_chunks} chunk(s) of up to "
              f"{batch_size:,} -> {n_chunks} array task(s) at {walltime} "
              "each.")
        if max_concurrent:
            print(f"Throttled to {max_concurrent} concurrent task(s).")

        prompt = "\nSubmit ligand prep -> docking array"
        prompt += " -> merge" if do_merge else ""
        prompt += " now?"
        if not assume_yes and not _confirm(prompt, default=True):
            print(f"Not submitting. Selected-molecule files were written "
                  f"to {self.final_dir}; job scripts were not generated.")
            return

        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.final_dir / "docked").mkdir(parents=True, exist_ok=True)

        ligprep_id = self._submit("final_ligprep",
                                   self._build_ligprep_script(),
                                   depends_on=None)
        array_id = self._submit(
            "final_docking_array",
            self._build_docking_array_script(n_chunks, batch_size, walltime,
                                              max_concurrent),
            depends_on=ligprep_id)
        merge_id = None
        if do_merge:
            merge_id = self._submit("final_docking_merge",
                                     self._build_merge_script(top_n),
                                     depends_on=array_id)

        self.state.data["final_docking"] = {
            "top_n": top_n, "batch_size": batch_size, "n_chunks": n_chunks,
            "walltime": walltime, "ligprep_job_id": ligprep_id,
            "array_job_id": array_id, "merge_job_id": merge_id,
            "submitted_at": str(datetime.now()),
        }
        self.state.save()

        print(f"\n{'=' * 60}")
        print("  Final docking submitted")
        print(f"  Ligand prep job:  {ligprep_id}")
        print(f"  Docking array:    {array_id}  (tasks 0-{n_chunks - 1})")
        if merge_id:
            print(f"  Merge job:        {merge_id}")
            print("  Handoff file (once merge completes):")
            print(f"    {self.final_dir}/final_top{top_n}_docked.sdf")
        print(f"  State log: {self.state.path}")
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive wizard: dock the top-N survivors of a "
                     "completed Deep Docking campaign as a Slurm job array. "
                     "A thin wrapper around dd_final_docking.py. See that "
                     "script for the select/prepare/dock/merge logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              python dd_final_docking_wizard.py --config campaign.yaml
              python dd_final_docking_wizard.py --config campaign.yaml --top-n 100000 --yes
              python dd_final_docking_wizard.py --config campaign.yaml --dry-run
        """),
    )
    parser.add_argument("--config", "-c", required=True,
                         help="Path to the campaign YAML config file")
    parser.add_argument("--iteration", type=int, default=None,
                         help="Treat this iteration as the final one "
                              "instead of auto-detecting it")
    parser.add_argument("--top-n", type=int, default=None,
                         help="Skip the interactive prompt and dock "
                              "this many top-scoring molecules")
    parser.add_argument("--batch-size", type=int, default=10000,
                         help="Molecules per array task (default: 10000)")
    parser.add_argument("--walltime", default=None,
                         help="Per-array-task walltime. Defaults to this "
                              "campaign's scheduler.walltime.phase3_docking.")
    parser.add_argument("--max-concurrent", type=int, default=None,
                         help="Throttle the array to this many simultaneous "
                              "tasks (SLURM '%%N' syntax). Unset = no limit.")
    parser.add_argument("--no-merge", dest="do_merge", action="store_false",
                         default=True,
                         help="Skip generating the post-array merge job")
    parser.add_argument("--yes", "-y", dest="assume_yes", action="store_true",
                         default=False,
                         help="Skip confirmation prompts (for scripted runs)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                         help="Print job scripts without submitting")
    args = parser.parse_args()

    if args.top_n is not None and args.top_n < 1:
        parser.error("--top-n must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    cfg = load_config(args.config)
    if cfg["scheduler"]["type"].upper() != "SLURM":
        parser.error("This wizard's job-array step only supports Slurm "
                      f"(campaign is configured for "
                      f"{cfg['scheduler']['type']}).")

    wizard = FinalDockingWizard(cfg, args.config, dry_run=args.dry_run)
    wizard.run(args.top_n, args.batch_size, args.walltime,
               args.max_concurrent, args.assume_yes, args.do_merge,
               iteration=args.iteration)


if __name__ == "__main__":
    main()
