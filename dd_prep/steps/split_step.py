"""
steps/split_step.py — Split the SMILES library into fixed-size chunk files.

Rationale:
─────────────────
The DD protocol is designed around a parallelised cluster workflow where
each node processes an independent subset of the library.  Chunking also
prevents any single OpenEye process from consuming the full library's RAM.
The DD paper recommends ~10 M molecules per chunk; this default can be
lowered for workstations with less memory.
─────────────────

Naming convention
─────────────────
Chunks are named ``smiles_all_<NNN>.smi`` with zero-padded indices wide
enough for the total count (e.g. 12 chunks → ``_001`` … ``_012``).
This convention is hardcoded in DD's downstream scripts, so it must be
preserved exactly.
─────────────────

Input  (from ctx):  "filter_file" if filter ran, else "input_file"
Output (to ctx):    "chunk_files"       — list of Path objects
                    "n_molecules_split" — total molecule count
"""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from dd_prep.config import SplitConfig
from dd_prep.steps.base import PipelineContext, PipelineStep
from dd_prep.utils.file_utils import remove_if_output_ready, zero_pad

logger = logging.getLogger(__name__)


class SplitStep(PipelineStep):
    name = "split"
    description = "Split library into fixed-size chunk files"

    def __init__(self, config: SplitConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        source = ctx.get("filter_file") or ctx.get("input_file", "") 
        if not source or not Path(source).is_file(): # Check the filter output first since it's preferred; fall back to raw input.
            errors.append(
                f"Source file not found: '{source}'. "
                "Set input_file in config or enable the filter step."
            )
        if self.config.chunk_size < 1: 
            errors.append(f"chunk_size must be >= 1, got {self.config.chunk_size}.")
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        cfg: SplitConfig = self.config

        cleanup: bool = ctx.get("cleanup_intermediates", False)

        # Prefer the filtered library if the filter step ran; fall back to raw.
        # Track whether we're consuming the filtered intermediate (which may be
        # deleted afterwards) versus the user's raw input (which never is).
        filter_file = ctx.get("filter_file")
        source = Path(filter_file or ctx.require("input_file"))
        out_dir = self._mkdir(ctx.work_dir / "smiles")

        # ── Resume check ──────────────────────────────────────────────────────
        existing = sorted(out_dir.glob("smiles_all_*.smi"))
        if existing:
            self.logger.info(
                "Resuming — found %d existing chunk files in %s.", len(existing), out_dir
            )
            ctx.set("chunk_files", existing)
            return ctx

        # ── Estimate zero-pad width from file size (no full second read) ──────
        # We estimate the chunk count from the file size and a sampled average
        # line length, then add one digit of headroom so the width is
        # comfortably large enough to keep names consistently sortable.
        pad_width = self._estimate_pad_width(source, cfg.chunk_size)
        self.logger.info(
            "  Streaming %s → chunks of up to %d (zero-pad width %d) …",
            source, cfg.chunk_size, pad_width,
        )

        # ── Stream through source, writing one chunk at a time ────────────────
        chunk_files: list[Path] = [] # Keep track of created chunk files to return in context.
        chunk_idx   = 0 # Current chunk index, starting at 0, incremented each time we start a new chunk.
        mol_count   = 0 # Total molecule count across all chunks, used for logging and context;
        current_fh  = None # File handle for the currently open chunk file, keeping it open while writing to avoid overhead (opening and closing).
        header_line = None # Store the header line to write it to each chunk; we read it once at the start and reuse it.

        # Open the source file and iterate through it line by line. tqdm provides a progress bar
        # that updates with each line, showing how many lines have been processed.
        with open(source) as src:
            for line_no, line in enumerate(
                tqdm(src, desc="Splitting", unit=" lines") #tqdm takes as an input an iterable (src file here). Wraps and yields the same lines, keeping track of how many have been yielded so far.
            ): # The enumerate object provides a line number for identifying header and logging.
                # First line is always the header; copy it to every chunk.
                if line_no == 0:
                    header_line = line
                    continue

                # Open a new chunk file when needed.
                if mol_count % cfg.chunk_size == 0:
                    if current_fh is not None: # Close the previous chunk file before starting a new one, if it's open.
                        current_fh.close()
                    chunk_idx += 1
                    chunk_name = out_dir / f"smiles_all_{str(chunk_idx).zfill(pad_width)}.smi" # Name the chunk file with the estimated zero-pad width to keep names consistently sortable.
                    current_fh = open(chunk_name, "w") # Open the new chunk file for writing; keep it open while we write lines to it.
                    current_fh.write(header_line)
                    chunk_files.append(chunk_name)

                current_fh.write(line) # Write the current line to the current chunk file.
                mol_count += 1

        if current_fh is not None:
            current_fh.close() # Close the last chunk file after finishing writing.

        self.logger.info(
            "  Split complete: %d files, %d molecules total.",
            len(chunk_files), mol_count,
        )

        # ── Optional cleanup ──────────────────────────────────────────────────
        # The chunk files now hold the full library, so the filtered
        # intermediate is dead weight. Delete it if cleanup is enabled and we
        # actually consumed the filtered file (never the user's raw input).
        # Guarded on having written at least one chunk so a zero-output run
        # can't silently destroy the only surviving copy.
        if cleanup and filter_file and chunk_files:
            src_path = Path(filter_file)
            try:
                src_path.unlink(missing_ok=True)
                self.logger.info(
                    "  cleanup_intermediates: removed filtered library %s", src_path
                )
            except OSError as exc:
                self.logger.warning("  Could not remove %s: %s", src_path, exc)

        ctx.set("chunk_files", chunk_files) # Store the list of created chunk files in the context for downstream steps to access.
        ctx.set("n_molecules_split", mol_count) # Store the total molecule count in the context for downstream steps to access and for logging.
        return ctx

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_pad_width(source: Path, chunk_size: int) -> int:
        """
        Estimate a zero-pad width wide enough for all chunk indices, without a
        full pass over the file.

        Samples the average bytes-per-line from the first portion of the file,
        divides the total file size by it to estimate the line count, converts
        that to an estimated chunk count, and returns the number of digits plus
        one of headroom (floor of MIN_WIDTH). Over-estimating is harmless (a few
        extra leading zeros), and the +1 headroom keeps names consistent even if the
        real count is somewhat larger than estimated.
        """
        MIN_WIDTH = 3
        SAMPLE_LINES = 1000
        try:
            total_bytes = source.stat().st_size
        except OSError:
            return MIN_WIDTH

        sampled_bytes = 0
        sampled_lines = 0
        with open(source, "rb") as fh:
            for _ in range(SAMPLE_LINES):
                line = fh.readline()
                if not line:
                    break
                sampled_bytes += len(line)
                sampled_lines += 1

        if sampled_lines == 0 or sampled_bytes == 0:
            return MIN_WIDTH

        avg_line = sampled_bytes / sampled_lines
        est_lines = max(1, int(total_bytes / avg_line))
        est_chunks = max(1, -(-est_lines // chunk_size))  # ceiling division
        return max(MIN_WIDTH, len(str(est_chunks)) + 1)