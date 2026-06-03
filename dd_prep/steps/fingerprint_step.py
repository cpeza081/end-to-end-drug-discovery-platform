"""
steps/fingerprint_step.py — RDKit Morgan fingerprint calculation.

Output format
─────────────
DD does not store fingerprints as dense binary vectors.  Instead it uses
a sparse format where only the indices of set (=1) bits are recorded:

    ZINC000001234567,3,47,112,398,512,701

This saves roughly 10× disk space for typical drug-like molecules whose
1024-bit fingerprints have ~50–80 set bits out of 1024.

The format is required as written by DD's ``sampling.py`` and
``extracting_morgan.py`` scripts.  The parameters (radius=2, nBits=1024)
must match those used when the DD model was trained. Changing them
produces incompatible feature vectors.

Parallelism
───────────
Each file is processed by a separate worker process via
``multiprocessing.Pool``.  This is CPU-bound work (RDKit hashing), so
real processes rather than threads give a genuine speedup.

The worker function ``_compute_fp_file`` must be a module-level function
(not a method or nested function) because multiprocessing pickles the
callable to send it to child processes.

Input  (from ctx):  "prepared_files" — list of Paths in library_prepared/
Output (to ctx):    "fp_files"       — list of Paths in library_prepared_fp/
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm

from dd_prep.config import FingerprintConfig
from dd_prep.steps.base import PipelineContext, PipelineStep


class FingerprintStep(PipelineStep):
    name = "fingerprint"
    description = "RDKit Morgan FP calculation → library_prepared_fp/"

    def __init__(self, config: FingerprintConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            errors.append("RDKit is required for the fingerprint step: pip install rdkit")
        if not ctx.get("prepared_files"):
            errors.append(
                "No prepared files in context. Ensure the organize step ran first."
            )
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # The main logic is in the module-level function _compute_fp_file, which is called in parallel by multiprocessing.Pool.  This method just sets up the job list and collects results.
        cfg: FingerprintConfig = self.config
        prepared_files: list[Path] = ctx.require("prepared_files")
        resume: bool = ctx.get("resume", True)
        dry_run: bool = ctx.get("dry_run", False)
        out_dir = self._mkdir(ctx.work_dir / "library_prepared_fp")

        # Build job list: (in_path, out_path, radius, n_bits)
        jobs: list[tuple[Path, Path, int, int]] = []
        fp_files: list[Path] = []

        # Check for existing output files to skip if resuming.
        for prep in prepared_files:
            out = out_dir / prep.name
            fp_files.append(out)
            if resume and out.is_file():
                self.logger.debug("Skipping %s — FP file already exists.", prep.name)
                continue
            jobs.append((prep, out, cfg.radius, cfg.n_bits))

        # If there are no jobs to run, we skip the parallel execution and return the context immediately.
        if not jobs:
            self.logger.info("All fingerprint files already present — skipping.")
            ctx.set("fp_files", fp_files)
            return ctx

        # If dry_run is True, we log which operations are to be executed, but we do not actually run them.
        if dry_run:
            for prep, out, _, _ in jobs:
                self.logger.info("[DRY RUN]  morgan_fp %s → %s", prep.name, out.name)
            ctx.set("fp_files", fp_files)
            return ctx

        # We log the number of files to process and the parameters for fingerprinting.
        self.logger.info(
            "Computing Morgan FPs (radius=%d, %d bits) for %d file(s) using %d worker(s) …",
            cfg.radius, cfg.n_bits, len(jobs), cfg.n_workers,
        )

        # Use 'spawn' context to avoid RDKit fork-safety issues on some platforms. "spawn" refers to the method of starting child processes where a fresh Python interpreter is launched, and only the necessary resources are passed. More robust than "fork".
        ctx_mp = mp.get_context("spawn") 
        with ctx_mp.Pool(processes=cfg.n_workers) as pool: # Create a multiprocessing pool with worker processes.
            list(tqdm(
                pool.imap(_compute_fp_file, jobs), # Map the _compute_fp_file function over the job list in parallel, returning an iterator of results.
                total=len(jobs),
                desc="Fingerprints",
                unit="file",
                ncols=80,
            )) # tqdm for a progress bar.

        ctx.set("fp_files", fp_files)
        return ctx


# ── Worker function (module-level for pickling) ───────────────────────────────

def _compute_fp_file(args: tuple[Path, Path, int, int]) -> None:
    """
    Compute Morgan fingerprints for every molecule in one SMILES file
    and write the sparse bit-index representation to an output file.

    This function runs in a child process — imports are deliberately local
    so the parent process does not need RDKit loaded at import time.

    Parameters
    ----------
    args : tuple
        (input_path, output_path, radius, n_bits)
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    in_path, out_path, radius, n_bits = args
    skipped = 0

    # We use RDKit here to parse the input SMILES and compute the Morgan fingerprint. We skip malformed lines.
    with open(in_path) as fh_in, open(out_path, "w") as fh_out:

        
        for line in fh_in:
            # Strip whitespace and skip empty lines.
            line = line.strip()
            if not line:
                continue
            
            # We split each line into parts and check that there are at least two. Part 1 is the SMILES string, part 2 is the molecule name.
            parts = line.split()
            if len(parts) < 2:
                skipped += 1
                continue

            # We use RDKit to parse the SMILES string into a molecule. If parsing fails, we skip the line.
            smiles, mol_name = parts[0], parts[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue

            # We compute the Morgan fingerprint as a bit vector
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits
            )
            set_bits = fp.GetOnBits() # get the indices of the set bits
            fh_out.write(f"{mol_name},{','.join(map(str, set_bits))}\n") # write the molecule name and indices to the output file.