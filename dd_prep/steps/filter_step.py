"""
steps/filter_step.py - RDKit-based property pre-filter.

Applies the same physicochemical filters as the original extract_smiles.py
(sLogP, RotBonds, MW, FSP3, ring counts, formal charge) but exposes every
threshold as a YAML-configurable parameter so they can be tightened,
relaxed, or disabled per project without touching the code.

Why pre-filter?
-----------------
Ultra-large libraries (1B+ molecules) contain many entries that would
trivially fail docking pharmacophore matching.  Discarding them before the
expensive OpenEye enumeration and fingerprinting steps can reduce the
library size by 30-70%, directly translating to shorter wall-clock times.

The filter is optional (enabled: false in YAML) for libraries that are
already curated (e.g. ZINC drug-like subset).
-----------------

How a new step is added:
-----------------
1. Create a new class subclassing PipelineStep, set name, implement run()
   and optionally validate().
2. Add a dataclass with enabled set to True to config.py and a default
   instance field to PipelineConfig.
3. Register it in pipeline.py's step list.
-----------------

Filter-only runs
-----------------
``dd-prep --step filter --config my.yaml`` runs this step alone and prints a
summary block with the number of molecules remaining.  Use it to size a
library, or to sweep thresholds before committing to a full prep run.

Note that re-running with different thresholds requires ``--no-resume``
(or deleting filtered/library_filtered.smi); otherwise the existing output
is reused and the new thresholds are ignored.

Input  (from ctx):  "input_file"   - path to raw SMILES library
Output (to ctx):    "filter_file"  - path to filtered SMILES library
                    "n_molecules_raw"      - molecule count before filter
                                             (None when resumed from disk)
                    "n_molecules_filtered" - molecule count after filter
                    "n_molecules_invalid"  - unparseable SMILES dropped
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from dd_prep.steps.base import PipelineStep, PipelineContext
from dd_prep.config import FilterConfig

logger = logging.getLogger(__name__)


class FilterStep(PipelineStep):
    name = "filter"
    description = "RDKit property-based pre-filter (sLogP, MW, FSP3, rings, ...)"

    # Chunk size for streaming reads (rows pulled from disk per pandas chunk).
    # Decrease if jobs are OOM-killed.
    STREAM_CHUNK_SIZE = 5_000_000

    # Rows per unit of work handed to a worker process. Small enough to keep
    # all workers busy and pickling cheap, large enough that per-task overhead
    # (process dispatch, RDKit import already amortised) stays negligible.
    WORKER_BATCH_SIZE = 20_000

    def __init__(self, config: FilterConfig) -> None:
        super().__init__(config)
 
    # ---- Validation ----------------------------------------------------------
 
    def validate(self, ctx: PipelineContext) -> list[str]:
        """
        Two checks: RDKit availability (tested by import, not just pip list)
        and input file existence (catches config typos before processing).
        """
        errors: list[str] = []
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            errors.append("RDKit is required for the filter step: pip install rdkit")
        input_file = ctx.get("input_file", "")
        if not input_file or not Path(input_file).is_file():
            errors.append(f"Input file not found: '{input_file}'")
        return errors
 
    # ---- Execution -----------------------------------------------------------
 
    def run(self, ctx: PipelineContext) -> PipelineContext:
        cfg: FilterConfig = self.config # type hint for convenience; self.config is actually just a dict, but we know from the pipeline setup that it has the structure of FilterConfig, so this lets us access config parameters with dot notation and get autocompletion in IDEs.
        input_file = Path(ctx.require("input_file"))
        out_dir = self._mkdir(ctx.work_dir / "filtered") # each step gets its own subdirectory under the main work_dir, which is named after the step for clarity. The _mkdir helper creates it if it doesn't exist and returns the path.
        out_file = out_dir / "library_filtered.smi" 

        # ---- Resume check ----------------------------------------------------
        # If output already exists, populate context and return.
        # This pattern is identical in every step, so any interrupted pipeline
        # can be restarted at the failed step.
        # Honour the resume flag: with resume=false (--no-resume) an existing
        # output is overwritten. Without this check, re-running after editing
        # thresholds would reuse the old file and report stale counts.
        resume: bool = ctx.get("resume", True)
        if resume and out_file.is_file():
            self.logger.info(
                "Resuming -- filtered file already exists: %s", out_file
            )
            n_filt = sum(1 for _ in out_file.open()) - 1  # subtract header
            ctx.set("filter_file", out_file)
            ctx.set("n_molecules_filtered", n_filt)
            ctx.set("n_molecules_raw", None)
            ctx.set("n_molecules_invalid", None)
            # Raw / invalid counts are not recoverable from the output file,
            # so the summary reports what is known and says so.
            self._log_summary(
                input_file=input_file,
                out_file=out_file,
                n_raw=None,
                n_invalid=None,
                n_passed=n_filt,
                resumed=True,
            )
            return ctx
 
        # ---- Detect file format from first line only -------------------------
        # Column detection runs on the first line so we never load the
        # full file into memory.
        sep, smiles_col, id_col = self._detect_format(input_file)
        self.logger.info(
            "  Detected format: sep=%r  smiles_col=%r  id_col=%r",
            sep, smiles_col, id_col,
        )
 
        # ---- Stream through file in fixed-size chunks ------------------------
        # Each pandas chunk is read from disk, then its rows are fanned out to a
        # pool of worker processes that do the RDKit parsing, descriptor
        # calculation, and threshold test. RDKit work is CPU-bound and
        # single-threaded per molecule, so without this pool the whole step runs
        # on one core regardless of --cpus-per-task, the dominant cost at
        # billion-molecule scale. n_workers=1 keeps serial behaviour.
        n_workers = max(1, int(getattr(cfg, "n_workers", 1)))
        thresholds = _thresholds_from_config(cfg)

        self.logger.info(
            "  Streaming %s in chunks of %d molecules using %d worker(s) ...",
            input_file, self.STREAM_CHUNK_SIZE, n_workers,
        )

        n_raw = n_invalid = n_passed = 0

        reader = pd.read_csv(
            input_file,
            sep=sep,
            engine="python",
            skipinitialspace=True,
            chunksize=self.STREAM_CHUNK_SIZE,
            usecols=[smiles_col, id_col],  # skip extra columns (e.g. Enamine
                                            # catalog fields) at read time
        )

        # Use a 'spawn' pool (fork is unsafe with RDKit on some platforms).
        pool = None
        mapper = map  # serial default
        if n_workers > 1:
            ctx_mp = mp.get_context("spawn")
            pool = ctx_mp.Pool(processes=n_workers)
            # imap keeps memory bounded (results streamed instead of materialised) and
            # preserves input order so output is deterministic.
            mapper = lambda fn, it: pool.imap(fn, it)

        try:
            with open(out_file, "w") as out_fh:
                out_fh.write("smiles idnumber\n")  # header, exactly once

                for chunk_idx, chunk in enumerate(reader):
                    # Standardise column names
                    chunk.columns = [c.strip().lower() for c in chunk.columns]
                    chunk = chunk.rename(
                        columns={smiles_col: "smiles", id_col: "idnumber"}
                    ).fillna("")

                    n_raw += len(chunk)

                    # Split this pandas chunk into small row batches and process
                    # them across the worker pool. Each batch returns the passing
                    # "smiles idnumber" lines plus (raw, invalid, passed) counts.
                    rows = list(zip(chunk["smiles"].tolist(),
                                    chunk["idnumber"].tolist()))
                    batches = [
                        (rows[i:i + self.WORKER_BATCH_SIZE], thresholds)
                        for i in range(0, len(rows), self.WORKER_BATCH_SIZE)
                    ]

                    for lines, (b_raw, b_invalid, b_passed) in mapper(
                        _filter_batch, batches
                    ):
                        if lines:
                            out_fh.write("".join(lines))
                        n_invalid += b_invalid
                        n_passed += b_passed

                    # Progress log every 10 chunks (every 5M molecules at default
                    # chunk size) so long runs aren't silent
                    if (chunk_idx + 1) % 10 == 0:
                        self.logger.info(
                            "  ... %d molecules processed, %d passed so far",
                            n_raw, n_passed,
                        )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if n_invalid:
            self.logger.warning(
                "  %d molecules had unparseable SMILES and were dropped.",
                n_invalid,
            )

        self._log_summary(
            input_file=input_file,
            out_file=out_file,
            n_raw=n_raw,
            n_invalid=n_invalid,
            n_passed=n_passed,
            resumed=False,
        )

        ctx.set("filter_file", out_file)
        ctx.set("n_molecules_filtered", n_passed)
        ctx.set("n_molecules_raw", n_raw)
        ctx.set("n_molecules_invalid", n_invalid)
        return ctx
 
    # ---- Helpers -------------------------------------------------------------

    def _log_summary(
        self,
        input_file: Path,
        out_file: Path,
        n_raw: int | None,
        n_invalid: int | None,
        n_passed: int,
        resumed: bool,
    ) -> None:
        """
        Emit the end-of-step summary block.

        Goes through the logger, so it lands both on the console (stdout) and
        in work_dir/dd_prep.log. The headline number is "molecules remaining",
        which is what a filter-only run is usually asking for.

        n_raw / n_invalid are None on a resumed run: those counts live only in
        the original run's log, not in the output file, so they are reported
        as unavailable.
        """
        log = self.logger.info
        bar = "-" * 58

        log(bar)
        log("  FILTER SUMMARY%s", "  (resumed from existing output)" if resumed else "")
        log("    Input                : %s", input_file)
        if n_raw is not None:
            log("    Molecules read       : %s", f"{n_raw:,}")
        else:
            log("    Molecules read       : n/a (see original run log)")
        if n_invalid:
            log("    Unparseable, dropped : %s", f"{n_invalid:,}")
        log("    MOLECULES REMAINING  : %s", f"{n_passed:,}")
        if n_raw:
            log("    Pass rate            : %.2f %%", 100 * n_passed / n_raw)
        log("    Output               : %s", out_file)
        log(bar)

    @staticmethod
    def _detect_format(path: Path) -> tuple[str, str, str]:
        """
        Detect separator, SMILES column name, and ID column name by reading
        only the first line of the file.
 
        Returns (sep, smiles_col, id_col) using the lowercased header names
        exactly as they appear in the file, so they can be passed directly
        to pd.read_csv(usecols=...).
        """
        from rdkit import Chem
 
        with open(path) as fh:
            first_line = fh.readline().strip()
            second_line = fh.readline().strip()  # one data row for fallback
 
        # Detect separator.
        # CRITICAL: must be a literal character, not a regex like r"\s+".
        # When sep is a regex, pandas reads the entire file into memory before
        # chunking, which defeats the purpose of chunksize entirely and causes
        # OOM kills on large libraries.  We detect the character used
        # and pass that literal string so pandas can stream efficiently.
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        elif " " in first_line.strip():
            sep = " "
        else:
            sep = "\t"  # safe fallback -- better than a regex
 
        # Split header into column names using any whitespace/delimiter
        import re
        cols = [c.strip().lower()
                for c in re.split(r"[\t, ]+", first_line)]
 
        smiles_names = {"smiles", "smi", "smile", "canonical_smiles"}
        id_names     = {"id", "idnumber", "name", "molecule_name",
                        "chembl_id", "zinc_id", "molid"}
 
        # Try to identify columns by name
        smiles_col = next((c for c in cols if c in smiles_names), None)
        id_col     = next((c for c in cols if c in id_names), None)
 
        if smiles_col and id_col:
            return sep, smiles_col, id_col
 
        # Fall back: try parsing the first data cell with RDKit
        if second_line:
            data_cols = re.split(r"[\t, ]+", second_line)
            if len(data_cols) >= 2:
                if Chem.MolFromSmiles(data_cols[0].strip()) is not None:
                    return sep, cols[0], cols[1]
                else:
                    return sep, cols[1], cols[0]
 
        # Last resort: assume first two columns are smiles, id
        return sep, cols[0], cols[1]


# ---- Parallel worker (module-level so it can be pickled by 'spawn') ----------

def _thresholds_from_config(cfg: FilterConfig) -> dict:
    """
    Flatten the threshold fields of a FilterConfig into a plain dict.

    Passed to each worker so children don't need to import the config module
    or unpickle a dataclass. Just a small dict of floats/ints.
    """
    return {
        "slogp_min":       cfg.slogp_min,
        "slogp_max":       cfg.slogp_max,
        "rot_bonds_max":   cfg.rot_bonds_max,
        "mw_min":          cfg.mw_min,
        "mw_max":          cfg.mw_max,
        "fsp3_min":        cfg.fsp3_min,
        "aro_rings_min":   cfg.aro_rings_min,
        "aro_rings_max":   cfg.aro_rings_max,
        "aliph_rings_max": cfg.aliph_rings_max,
        "total_rings_min": cfg.total_rings_min,
        "total_rings_max": cfg.total_rings_max,
        "formal_charge":   cfg.formal_charge,
        "chiral_centers_max": getattr(cfg, "chiral_centers_max", None),
    }


def _filter_batch(
    args: tuple[list[tuple[str, str]], dict]
) -> tuple[list[str], tuple[int, int, int]]:
    """
    Filter one batch of ``(smiles, idnumber)`` rows.

    Runs in a worker process. Imports RDKit locally so the parent never needs
    it loaded. For each molecule it parses the SMILES, then applies the eight
    physicochemical thresholds with short-circuit evaluation (cheapest checks
    first, bail on the first failure). This is faster than computing every descriptor
    for every molecule the way a full-DataFrame mask does.

    Parameters
    ----------
    args : (rows, thresholds)
        rows       : list of (smiles, idnumber) tuples
        thresholds : dict from _thresholds_from_config

    Returns
    -------
    (lines, (n_raw, n_invalid, n_passed))
        lines : list of "smiles idnumber\\n" strings for molecules that passed
        counts: batch-local totals for aggregation by the parent
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops

    rows, th = args
    lines: list[str] = []
    n_invalid = 0

    # Decide once per batch which descriptors are actually needed. A threshold
    # of None disables its check, and we then skip computing the descriptor
    # entirely rather than computing it and ignoring the result.
    do_slogp  = th["slogp_min"] is not None or th["slogp_max"] is not None
    do_rot    = th["rot_bonds_max"] is not None
    do_mw     = th["mw_min"] is not None or th["mw_max"] is not None
    do_fsp3   = th["fsp3_min"] is not None
    do_aro    = th["aro_rings_min"] is not None or th["aro_rings_max"] is not None
    do_aliph  = th["aliph_rings_max"] is not None
    do_total  = th["total_rings_min"] is not None or th["total_rings_max"] is not None
    do_charge = th["formal_charge"] is not None
    do_chiral = th["chiral_centers_max"] is not None
    # Ring counts feed the total-rings test, so they may be needed even when
    # their own individual thresholds are off.
    need_aro   = do_aro or do_total
    need_aliph = do_aliph or do_total

    for smiles, idnumber in rows:
        s = str(smiles) if smiles is not None else ""
        if not s:
            n_invalid += 1
            continue

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_invalid += 1
            continue

        # Short-circuit threshold tests, cheapest / most-selective first.
        if do_slogp and _outside(Descriptors.MolLogP(mol),
                                 th["slogp_min"], th["slogp_max"]):
            continue
        if do_rot and Descriptors.NumRotatableBonds(mol) > th["rot_bonds_max"]:
            continue
        if do_mw and _outside(Descriptors.ExactMolWt(mol),
                              th["mw_min"], th["mw_max"]):
            continue
        if do_fsp3 and rdMolDescriptors.CalcFractionCSP3(mol) < th["fsp3_min"]:
            continue

        if need_aro:
            aro = rdMolDescriptors.CalcNumAromaticRings(mol)
            if do_aro and _outside(aro, th["aro_rings_min"], th["aro_rings_max"]):
                continue
        if need_aliph:
            aliph = rdMolDescriptors.CalcNumAliphaticRings(mol)
            if do_aliph and aliph > th["aliph_rings_max"]:
                continue
        if do_total and _outside(aro + aliph,
                                 th["total_rings_min"], th["total_rings_max"]):
            continue

        if do_charge and rdmolops.GetFormalCharge(mol) != th["formal_charge"]:
            continue

        # Stereocentre cap goes last: it is the most expensive remaining test
        # (roughly the cost of parsing the molecule again), so it only runs on
        # molecules that already survived everything else.
        if do_chiral and _count_stereocentres(mol) > th["chiral_centers_max"]:
            continue

        lines.append(f"{s} {idnumber}\n")

    return lines, (len(rows), n_invalid, len(lines))


def _outside(value, lo, hi) -> bool:
    """
    True if *value* falls outside the closed interval [lo, hi].

    A bound of None is treated as unbounded on that side, which is how a
    half-open filter (e.g. "MW at most 450, no lower limit") is expressed.
    """
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


def _count_stereocentres(mol) -> int:
    """
    Number of tetrahedral stereocentres, counting specified and unspecified.

    Uses Chem.FindPotentialStereo, RDKit's current stereo perception. It
    reports every potential stereo element with a .specified flag, so one
    call covers both declared (@ / @@) and undeclared centres.

    Deliberately NOT implemented as
        CalcNumAtomStereoCenters + CalcNumUnspecifiedAtomStereoCenters
    which is the obvious-looking approach but double-counts: once
    AssignStereochemistry has run with flagPossibleStereoCenters, the
    unspecified centres are included in both terms. That overcounts an
    all-undeclared molecule by exactly 2x.

    Double-bond (E/Z) stereo is excluded. Only Atom_Tetrahedral elements
    are counted.
    """
    from rdkit import Chem

    return sum(
        1 for element in Chem.FindPotentialStereo(mol)
        if element.type == Chem.StereoType.Atom_Tetrahedral
    )
