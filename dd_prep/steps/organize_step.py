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

This step places each processed file into ``library_prepared/`` under its
canonical name.  How the file is placed is controlled by ``organize.mode``:

    "hardlink" (default) create a hard link: no bytes are copied, no extra
                           disk is used, and the intermediate is preserved for
                           resume.  Falls back to a copy if the destination is
                           on a different filesystem.
    "move"               rename the file into place: no extra disk, and the
                           intermediate is consumed.
    "copy"               physically copy (the original behaviour): doubles
                           this stage's disk footprint.

At TB scale a full copy of every chunk is one of the largest contributors to
project-space usage, so hardlink/move are strongly preferred.

Graceful fallback
─────────────────
OrganizeStep checks for ``state_files`` first (tautomers ran), then
``isom_files`` (only flipper ran), then ``chunk_files`` (neither ran).
This means any combination of enabled/disabled upstream steps works.

Input  (from ctx):  "state_files" → "isom_files" → "chunk_files" (priority)
Output (to ctx):    "prepared_files" — list of Paths in library_prepared/
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dd_prep.config import OrganizeConfig
from dd_prep.steps.base import PipelineContext, PipelineStep

_VALID_MODES = ("hardlink", "move", "copy")


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
        errors: list[str] = []
        if not has_input:
            errors.append(
                "No processed files found in context. "
                "Ensure at least the split step has run."
            )
        mode = getattr(self.config, "mode", "hardlink")
        if mode not in _VALID_MODES:
            errors.append(
                f"organize.mode must be one of {_VALID_MODES}, got {mode!r}."
            )
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        resume: bool = ctx.get("resume", True)
        mode: str = getattr(self.config, "mode", "hardlink")
        out_dir = self._mkdir(ctx.work_dir / "library_prepared")

        # Pick the most-processed set of files available.
        source_files: list[Path] = (
            ctx.get("state_files")
            or ctx.get("isom_files")
            or ctx.require("chunk_files")
        )

        prepared_files: list[Path] = []
        placed = 0

        for src in source_files:
            dest_name = _canonical_name(src)
            dest = out_dir / dest_name
            prepared_files.append(dest)

            if resume and dest.is_file():
                self.logger.debug("Skipping %s — already in library_prepared/.", dest_name)
                continue

            used_mode = self._place(src, dest, mode)
            self.logger.debug(
                "  %s  →  library_prepared/%s  (%s)", src.name, dest_name, used_mode
            )
            placed += 1

        self.logger.info(
            "  %d file(s) placed into library_prepared/ via '%s' (%d already present).",
            placed,
            mode,
            len(prepared_files) - placed,
        )

        ctx.set("prepared_files", prepared_files)
        return ctx

    # ── File placement ────────────────────────────────────────────────────────

    @staticmethod
    def _place(src: Path, dest: Path, mode: str) -> str:
        """
        Put *src* at *dest* using *mode*.  Returns the mode actually used
        (may differ from the request if a hard link falls back to a copy).

        Idempotent w.r.t. a stale/partial dest: any existing dest is removed
        first so hardlink/move don't raise FileExistsError.
        """
        if dest.exists():
            dest.unlink()

        if mode == "copy":
            shutil.copy2(src, dest)
            return "copy"

        if mode == "move":
            # shutil.move handles cross-device moves (copy + unlink) internally.
            shutil.move(str(src), str(dest))
            return "move"

        # mode == "hardlink": try a hard link, fall back to a copy if the
        # destination is on a different filesystem (EXDEV) or the OS refuses.
        try:
            os.link(src, dest)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dest)
            return "copy (hardlink unavailable)"


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