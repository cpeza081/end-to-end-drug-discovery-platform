"""
steps/tautomer_step.py — OpenEye TAUTOMERS protonation state assignment.

TAUTOMERS function:
───────────────────
TAUTOMERS (part of the OpenEye QUACPAC toolkit) enumerates tautomers of
each molecule and returns the most stable form at a given pH.  With
``-maxtoreturn 1`` it returns only the dominant tautomer — the single most
likely protonation state at physiological pH (7.4).

-ch3 false flag:
───────────────
The ``-ch3 false`` flag prevents TAUTOMERS from considering methyl group
tautomerism, which can produce chemically unreasonable structures near
heteroatom-containing rings.  This is the setting recommended in the VS
preparation SOP for this pipeline.

Graceful fallback
─────────────────
If the flipper step was disabled, ``isom_files`` will not be in the
context.  TautomerStep automatically falls back to using ``chunk_files``
directly, so disabling flipper does not break the rest of the pipeline.

Input  (from ctx):  "isom_files"   (preferred) or "chunk_files" (fallback)
Output (to ctx):    "state_files"  — list of *_states.smi paths
"""

from __future__ import annotations

from pathlib import Path

from dd_prep.config import TautomerConfig
from dd_prep.steps.base import PipelineContext, PipelineStep
from dd_prep.utils.parallel import run_parallel


class TautomerStep(PipelineStep):
    name = "tautomer"
    description = "OpenEye TAUTOMERS — dominant tautomer / protonation state"

    def __init__(self, config: TautomerConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors = self._check_binary("tautomers")
        if not ctx.get("isom_files") and not ctx.get("chunk_files"):
            errors.append(
                "No input files in context. "
                "Ensure either the flipper or the split step ran first."
            )
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        cfg: TautomerConfig = self.config
        resume: bool = ctx.get("resume", True)
        n_parallel: int = ctx.get("n_parallel", 4)
        dry_run: bool = ctx.get("dry_run", False)

        # Use isomer-expanded files if available; fall back to raw chunks.
        input_files: list[Path] = ctx.get("isom_files") or ctx.require("chunk_files")

        # In Slurm array mode, restrict to the one file for this task.
        chunk_index: int | None = ctx.get("chunk_index")
        if chunk_index is not None:
            if chunk_index >= len(input_files):
                self.logger.info(
                    "chunk_index=%d >= total files (%d) — nothing to do.",
                    chunk_index, len(input_files),
                )
                ctx.set("state_files", [])
                return ctx
            input_files = [input_files[chunk_index]]

        commands: list[list[str]] = []
        state_files: list[Path] = []

        for in_file in input_files:
            # Strip any intermediate suffixes, then append _states.
            base = in_file.name.replace("_isom.smi", "").replace(".smi", "")
            out = in_file.parent / f"{base}_states.smi"
            state_files.append(out)

            if resume and out.is_file():
                self.logger.debug("Skipping %s — state file already exists.", in_file.name)
                continue

            cmd = [
                "tautomers",
                "-in",           str(in_file),
                "-out",          str(out),
                "-maxtoreturn",  str(cfg.max_to_return),
            ]
            if not cfg.ch3:
                cmd.extend(["-ch3", "false"])
            if cfg.warts:
                cmd.append("-warts")
            if cfg.extra_args:
                cmd.extend(cfg.extra_args.split())
            commands.append(cmd)

        if commands:
            self.logger.info("Running TAUTOMERS on %d file(s) …", len(commands))
            run_parallel(commands, n_parallel, dry_run, desc="Tautomers")
        else:
            self.logger.info("All tautomer state files already present — skipping.")

        ctx.set("state_files", state_files)
        return ctx