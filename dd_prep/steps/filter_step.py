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

Input  (from ctx):  "input_file"   - path to raw SMILES library
Output (to ctx):    "filter_file"  - path to filtered SMILES library
                    "n_molecules_raw"      - molecule count before filter
                    "n_molecules_filtered" - molecule count after filter
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from dd_prep.steps.base import PipelineStep, PipelineContext
from dd_prep.config import FilterConfig

logger = logging.getLogger(__name__)


class FilterStep(PipelineStep):
    name = "filter"
    description = "RDKit property-based pre-filter (sLogP, MW, FSP3, rings, ...)"

    # Chunk size for streaming reads.
    # Decrease if jobs are OOM-killed.
    STREAM_CHUNK_SIZE = 1_000_000

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
        # Imported here so validate() runs first; if RDKit is missing the
        # error is contextual rather than a bare ImportError at startup.
        from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolops
        from rdkit import Chem

        cfg: FilterConfig = self.config # type hint for convenience; self.config is actually just a dict, but we know from the pipeline setup that it has the structure of FilterConfig, so this lets us access config parameters with dot notation and get autocompletion in IDEs.
        input_file = Path(ctx.require("input_file")) 
        out_dir = self._mkdir(ctx.work_dir / "filtered") # each step gets its own subdirectory under the main work_dir, which is named after the step for clarity. The _mkdir helper creates it if it doesn't exist and returns the path.
        out_file = out_dir / "library_filtered.smi" 

        # ---- Resume check ----------------------------------------------------
        # If output already exists, populate context and return.
        # This pattern is identical in every step, so any interrupted pipeline
        # can be restarted at the failed step.
        if out_file.is_file():
            self.logger.info(
                "Resuming -- filtered file already exists: %s", out_file
            )
            n_filt = sum(1 for _ in out_file.open()) - 1  # subtract header
            ctx.set("filter_file", out_file)
            ctx.set("n_molecules_filtered", n_filt)
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
        # Each chunk is loaded, filtered, and appended to the output file
        # before the next chunk is read.
        self.logger.info(
            "  Streaming %s in chunks of %d molecules ...",
            input_file, self.STREAM_CHUNK_SIZE,
        )

        n_raw = n_invalid = n_passed = 0
        header_written = False

        reader = pd.read_csv(
            input_file,
            sep=sep,
            engine="python",
            skipinitialspace=True,
            chunksize=self.STREAM_CHUNK_SIZE,
            usecols=[smiles_col, id_col],  # skip extra columns (e.g. Enamine
                                            # catalog fields) at read time
        )

        with open(out_file, "w") as out_fh:
            for chunk_idx, chunk in enumerate(reader):

                # Standardise column names
                chunk.columns = [c.strip().lower() for c in chunk.columns]
                chunk = chunk.rename(
                    columns={smiles_col: "smiles", id_col: "idnumber"}
                ).fillna("").copy()

                chunk_raw = len(chunk)
                n_raw += chunk_raw

                # Write header once at the top of the output file.
                if not header_written:
                    out_fh.write("smiles idnumber\n")
                    header_written = True

                # Parse SMILES, invalid entries become None and are dropped.
                chunk["mol"] = chunk["smiles"].apply(
                    lambda s: Chem.MolFromSmiles(str(s)) if s else None
                )
                chunk = chunk[chunk["mol"].notna()].copy()
                n_invalid += chunk_raw - len(chunk)

                # Compute descriptors for this chunk
                chunk["sLogP"]      = chunk["mol"].apply(Descriptors.MolLogP)
                chunk["RotBonds"]   = chunk["mol"].apply(Descriptors.NumRotatableBonds)
                chunk["MW"]         = chunk["mol"].apply(Descriptors.ExactMolWt)
                chunk["FSP3"]       = chunk["mol"].apply(rdMolDescriptors.CalcFractionCSP3)
                chunk["AroRings"]   = chunk["mol"].apply(rdMolDescriptors.CalcNumAromaticRings)
                chunk["AliphRings"] = chunk["mol"].apply(rdMolDescriptors.CalcNumAliphaticRings)
                chunk["TotRings"]   = chunk["AroRings"] + chunk["AliphRings"]
                chunk["Charge"]     = chunk["mol"].apply(rdmolops.GetFormalCharge)

                # Apply all filters in a single combined boolean mask.
                mask = (
                    chunk["sLogP"].between(cfg.slogp_min, cfg.slogp_max) &
                    (chunk["RotBonds"] <= cfg.rot_bonds_max) &
                    chunk["MW"].between(cfg.mw_min, cfg.mw_max) &
                    (chunk["FSP3"] >= cfg.fsp3_min) &
                    chunk["AroRings"].between(cfg.aro_rings_min, cfg.aro_rings_max) &
                    (chunk["AliphRings"] <= cfg.aliph_rings_max) &
                    chunk["TotRings"].between(cfg.total_rings_min, cfg.total_rings_max) &
                    (chunk["Charge"] == cfg.formal_charge)
                )
                passed = chunk[mask]
                n_passed += len(passed)

                # Append passing molecules to output
                passed[["smiles", "idnumber"]].to_csv(
                    out_fh, sep=" ", index=False, header=False
                )

                # Progress log every 10 chunks (every 5M molecules at default
                # chunk size)
                if (chunk_idx + 1) % 10 == 0:
                    self.logger.info(
                        "  ... %d molecules processed, %d passed so far",
                        n_raw, n_passed,
                    )

        if n_invalid:
            self.logger.warning(
                "  %d molecules had unparseable SMILES and were dropped.",
                n_invalid,
            )
        self.logger.info(
            "  Filter complete: %d / %d molecules passed (%.1f %%).",
            n_passed, n_raw, 100 * n_passed / max(n_raw, 1),
        )
        self.logger.info("  Written to %s", out_file)

        ctx.set("filter_file", out_file)
        ctx.set("n_molecules_filtered", n_passed)
        ctx.set("n_molecules_raw", n_raw)
        return ctx

    # ---- Helpers -------------------------------------------------------------

    @staticmethod
    def _detect_format(path: Path) -> tuple[str, str, str]:
        """
        Detect separator, SMILES column name, and ID column name by reading
        only the first line of the file.

        Returns (sep, smiles_col, id_col) using the lowercased header names
        as they appear in the file, so they can be passed directly to pd.read_csv(usecols=...).
        """
        from rdkit import Chem

        with open(path) as fh:
            first_line = fh.readline().strip()
            second_line = fh.readline().strip()  # one data row for fallback

        # Detect separator
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        else:
            sep = r"\s+"

        # Split header into column names
        import re
        cols = [c.strip().lower()
                for c in re.split(r"\t|,|\s+" if sep == r"\s+" else sep,
                                  first_line)]

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
            data_cols = re.split(r"\t|,|\s+" if sep == r"\s+" else sep,
                                 second_line)
            if len(data_cols) >= 2:
                if Chem.MolFromSmiles(data_cols[0].strip()) is not None:
                    return sep, cols[0], cols[1]
                else:
                    return sep, cols[1], cols[0]

        # Last resort: assume first two columns are smiles, id
        return sep, cols[0], cols[1]