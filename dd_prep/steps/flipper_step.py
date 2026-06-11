"""
steps/flipper_step.py — OpenEye FLIPPER stereoisomer enumeration.

What FLIPPER does
─────────────────
FLIPPER enumerates all unspecified stereocentres in each molecule,
producing one output entry per possible stereoisomer.  For a molecule
with two unspecified chiral centres this yields up to 4 isomers (2²).

Rationale
─────────────────────────────
Docking is stereospecific.  Running docking on a single arbitrary assignment would miss the
active enantiomer.  Enumerating first ensures all
geometrically plausible forms enter the screen.

The ``-warts`` flag appends ``_1``, ``_2``, … to each isomer name so that
all downstream files have unique identifiers (required by DD's sampling
and fingerprinting scripts).

Input  (from ctx):  "chunk_files"  — list of .smi chunk paths
Output (to ctx):    "isom_files"   — list of *_isom.smi paths (one per chunk)
"""

from __future__ import annotations

from pathlib import Path

from dd_prep.config import FlipperConfig
from dd_prep.steps.base import PipelineContext, PipelineStep
from dd_prep.utils.parallel import run_parallel


class FlipperStep(PipelineStep):
    name = "flipper"
    description = "OpenEye FLIPPER — enumerate unspecified stereocentres"

    def __init__(self, config: FlipperConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors = self._check_binary("flipper")
        if not ctx.get("chunk_files"):
            errors.append("No chunk files in context. Ensure the split step ran first.")
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        cfg: FlipperConfig = self.config
        chunk_files: list[Path] = ctx.require("chunk_files")
        resume: bool = ctx.get("resume", True)
        n_parallel: int = ctx.get("n_parallel", 4)
        dry_run: bool = ctx.get("dry_run", False)

        # In Slurm array mode, restrict to the one chunk for this task.
        chunk_index: int | None = ctx.get("chunk_index")
        if chunk_index is not None:
            if chunk_index >= len(chunk_files):
                self.logger.info(
                    "chunk_index=%d >= total chunks (%d). Nothing to do.",
                    chunk_index, len(chunk_files),
                )
                ctx.set("isom_files", [])
                return ctx
            chunk_files = [chunk_files[chunk_index]]

        commands: list[list[str]] = []
        isom_files: list[Path] = []

        for chunk in chunk_files:
            out = chunk.parent / chunk.name.replace(".smi", "_isom.smi")
            isom_files.append(out) # Collect the expected output paths so we can populate the context at the end, even if some files are skipped due to resume.

            if resume and out.is_file():
                self.logger.debug("Skipping %s — isomer file already exists.", chunk.name)
                continue

            cmd = ["flipper", "-in", str(chunk), "-out", str(out)]
            if cfg.warts: 
                cmd.append("-warts")
            if cfg.enum_nitrogen:
                cmd.append("-enumNitrogen")
            if cfg.extra_args:
                cmd.extend(cfg.extra_args.split())
            commands.append(cmd)

        if commands: # If there are any commands to run (i.e. any files that need processing), execute them in parallel; otherwise, skip and just populate the context with the existing files.
            self.logger.info("Running FLIPPER on %d chunk(s) …", len(commands)) 
            run_parallel(commands, n_parallel, dry_run, desc="Flipper") 
        else:
            self.logger.info("All isomer files already present — skipping FLIPPER.")

        ctx.set("isom_files", isom_files) 
        return ctx