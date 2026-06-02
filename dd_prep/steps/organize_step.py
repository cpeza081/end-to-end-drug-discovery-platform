"""
steps/organize_step.py — Collect processed files into library_prepared/.

What this step does
───────────────────
After tautomers (or flipper, or the raw split), the SMILES files live in
the ``smiles/`` working directory with names like:

    smiles_all_001_isom_states.smi   (full pipeline)
    smiles_all_001_isom.smi          (no tautomers)
    smiles_all_001.smi               (no flipper, no tautomers)

DD's downstream scripts (sampling.py, morgan_fp.py) expect their input
files in a flat directory called ``library_prepared/`` with names ending
in ``.txt``:

    library_prepared/smiles_all_001.txt

This step copies (to preserve intermediates) and renames each
file to satisfy that convention.

Graceful fallback
─────────────────
OrganizeStep checks for ``state_files`` first (tautomers ran), then
``isom_files`` (only flipper ran), then ``chunk_files`` (neither ran).
This means any combination of enabled/disabled upstream steps works.

Input  (from ctx):  "state_files" → "isom_files" → "chunk_files" (priority)
Output (to ctx):    "prepared_files" — list of Paths in library_prepared/
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dd_prep.config import OrganizeConfig
from dd_prep.steps.base import PipelineContext, PipelineStep


class OrganizeStep(PipelineStep):
    name = "organize"
    description = "Collect processed files into library_prepared/"

    def __init__(self, config: OrganizeConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        has_input = (
            ctx.get("state_files")
            or ctx.get("isom_files")
            or ctx.get("chunk_files")
        )
        if not has_input:
            return [
                "No processed files found in context. "
                "Ensure at least the split step has run."
            ]
        return []

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        resume: bool = ctx.get("resume", True)
        out_dir = self._mkdir(ctx.work_dir / "library_prepared")

        # Pick the most-processed set of files available.
        source_files: list[Path] = (
            ctx.get("state_files")
            or ctx.get("isom_files")
            or ctx.require("chunk_files")
        )

        prepared_files: list[Path] = []
        copied = 0

        for src in source_files:
            dest_name = _canonical_name(src)
            dest = out_dir / dest_name
            prepared_files.append(dest)

            if resume and dest.is_file():
                self.logger.debug("Skipping %s — already in library_prepared/.", dest_name)
                continue

            shutil.copy2(src, dest)
            self.logger.debug("  %s  →  library_prepared/%s", src.name, dest_name)
            copied += 1

        self.logger.info(
            "  %d file(s) copied to library_prepared/ (%d already present).",
            copied,
            len(prepared_files) - copied,
        )

        ctx.set("prepared_files", prepared_files)
        return ctx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _canonical_name(path: Path) -> str:
    """
    Strip all intermediate suffixes and return the canonical DD filename.

    Examples
    --------
    smiles_all_001_isom_states.smi  →  smiles_all_001.txt
    smiles_all_001_isom.smi         →  smiles_all_001.txt
    smiles_all_001.smi              →  smiles_all_001.txt
    """
    name = path.stem  # remove extension
    for suffix in ("_isom_states", "_isom", "_states"):
        name = name.replace(suffix, "")
    return name + ".txt"