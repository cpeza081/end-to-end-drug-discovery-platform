"""
steps/omega_step.py — OpenEye OMEGA 3-D conformer generation (optional).

Why this step is disabled by default and when it is appropriate to enable it.
────────────────────────
This step is NOT needed for the initial
library preparation.  The DD machine-learning model only needs 2-D Morgan
fingerprints (produced by FingerprintStep).

OMEGA is needed at Stage IV of the DD protocol, when the sampled molecules
from each iteration must be docked.  Enable this step only if you want the
entire library pre-processed in 3-D upfront (unusual for libraries > 10 M).

Mode "classic" vs "pose"
─────────────────────────
  classic  — one low-energy conformer per molecule; suitable for Glide SP/XP
  pose     — multi-conformer OEB; used as input for FRED docking

Output format
─────────────
  "sdf.gz"  — compressed SDF; required for Glide and most SBDD software
  "oeb.gz"  — OpenEye binary format; required for FRED

Both formats preserve the molecule name from the input SMILES file so
docking results can be traced back to their library entry.

Input  (from ctx):  "prepared_files" — Paths in library_prepared/
Output (to ctx):    "sdf_files"      — Paths in sdf/ or oeb/
"""

from __future__ import annotations

from pathlib import Path

from dd_prep.config import OmegaConfig
from dd_prep.steps.base import PipelineContext, PipelineStep
from dd_prep.utils.parallel import run_parallel


class OmegaStep(PipelineStep):
    name = "omega"
    description = "OpenEye OMEGA — 3-D conformer generation (optional)"

    def __init__(self, config: OmegaConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors = self._check_binary("oeomega")
        if not ctx.get("prepared_files"):
            errors.append(
                "No prepared files in context. Ensure the organize step ran first."
            )
        valid_modes = {"classic", "pose"}
        if self.config.mode not in valid_modes:
            errors.append(
                f"omega.mode must be one of {valid_modes}, got '{self.config.mode}'."
            )
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        cfg: OmegaConfig = self.config
        prepared_files: list[Path] = ctx.require("prepared_files")
        resume: bool = ctx.get("resume", True)
        n_parallel: int = ctx.get("n_parallel", 4)
        dry_run: bool = ctx.get("dry_run", False)

        # Output directory named after the format (sdf or oeb).
        out_ext = cfg.output_format.split(".")[0]   # "sdf" or "oeb"
        out_dir = self._mkdir(ctx.work_dir / out_ext)

        # In SLURM array mode, restrict to the one file for this task.
        chunk_index: int | None = ctx.get("chunk_index")
        if chunk_index is not None:
            if chunk_index >= len(prepared_files):
                self.logger.info(
                    "chunk_index=%d >= total files (%d) — nothing to do.",
                    chunk_index, len(prepared_files),
                )
                ctx.set("sdf_files", [])
                return ctx
            prepared_files = [prepared_files[chunk_index]]

        commands: list[list[str]] = []
        sdf_files: list[Path] = []

        for prep in prepared_files:
            out = out_dir / (prep.stem + "." + cfg.output_format)
            sdf_files.append(out)

            if resume and out.is_file():
                self.logger.debug("Skipping %s — 3-D file already exists.", prep.name)
                continue

            # Build the OMEGA command.
            cmd = [
                "oeomega", cfg.mode,
                "-in",          str(prep), # input file in library_prepared/
                "-out",         str(out), # output file in sdf/ or oeb/
                "-maxconfs",    str(cfg.max_confs), # maximum number of conformers per molecule; OMEGA may produce fewer if it can't find that many under the energy window
                "-mpi_np",      str(cfg.mpi_np), # number of parallel processes for OMEGA's internal MPI parallelization; set to 1 to disable OMEGA's internal parallelism and rely solely on the outer level of parallelism in run_parallel, which may be more efficient for large libraries on a cluster
                "-strictstereo", "true" if cfg.strict_stereo else "false", # whether to discard molecules with undefined or ambiguous stereochemistry.
            ]
            if cfg.extra_args: # any additional OMEGA command-line arguments specified by the user in the config allowing users to customize OMEGA's behavior beyond the parameters exposed in OmegaConfig.
                cmd.extend(cfg.extra_args.split())
            commands.append(cmd)

        if commands:
            self.logger.info(
                "Running OMEGA (%s mode) on %d file(s) …", cfg.mode, len(commands)
            )
            run_parallel(commands, n_parallel, dry_run, desc="Omega 3-D") # execute the OMEGA commands in parallel, respecting the n_parallel and dry_run settings from the context. The desc parameter is used to label the progress bar.
        else:
            self.logger.info("All 3-D files already present — skipping OMEGA.")

        ctx.set("sdf_files", sdf_files)
        return ctx