"""
steps/filter_step.py — RDKit-based property pre-filter.

Applies the same physicochemical filters as the original extract_smiles.py
(sLogP, RotBonds, MW, FSP3, ring counts, formal charge) but exposes every
threshold as a YAML-configurable parameter so they can be tightened,
relaxed, or disabled per project without touching the code.

Why pre-filter?
─────────────────────────────
Ultra-large libraries (1B+ molecules) contain many entries that would
trivially fail docking pharmacophore matching.  Discarding them before the
expensive OpenEye enumeration and fingerprinting steps can reduce the
library size by 30–70 %, directly translating to shorter wall-clock times.

The filter is optional (enabled: false in YAML) for libraries that are
already curated (e.g. ZINC drug-like subset).
─────────────────────────────

How a new step is added:
─────────────────────────────
1. Create a new class subclassing PipelineStep, set name, implement run() and optionally validate().
2. Add a dataclass with enabled set to True to config.py and a default instance field to PipelineConfig.
3. Register it in pipeline.py's step list. 
─────────────────────────────


Input  (from ctx):  "input_file"   — path to raw SMILES library
Output (to ctx):    "filter_file"  — path to filtered SMILES library
                    "n_molecules_raw"      — molecule count before filter
                    "n_molecules_filtered" — molecule count after filter
"""

from __future__ import annotations

import logging
from pathlib import Path

from dd_prep.steps.base import PipelineStep, PipelineContext
from dd_prep.config import FilterConfig

import pandas as pd

logger = logging.getLogger(__name__)


class FilterStep(PipelineStep):
    # Sets the two class-level identifiers. The name must match the key in PipelineConfig so the pipeline can pair them automatically.
    name = "filter"
    description = "RDKit property-based pre-filter (sLogP, MW, FSP3, rings, …)"

    def __init__(self, config: FilterConfig) -> None:
        super().__init__(config)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, ctx: PipelineContext) -> list[str]:
        """Two checks here. RDKit availability is tested by actually importing it rather than checking pip.
        An installed but broken package would pass a find_spec check but fail here. The second check is a
        file existence check to catch typos in the config path before any processing begins."""
        errors: list[str] = []
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            errors.append("RDKit is required for the filter step: pip install rdkit")
        input_file = ctx.get("input_file", "")
        if not input_file or not Path(input_file).is_file():
            errors.append(f"Input file not found: '{input_file}'")
        return errors

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, ctx: PipelineContext) -> PipelineContext:
        from rdkit.Chem import PandasTools, Descriptors, rdMolDescriptors, rdmolops # imported here to ensure the validate() check ran first; if RDKit isn't available, the step will fail during validation rather than after hours of processing.

        cfg: FilterConfig = self.config # type hint for convenience; self.config is actually just a dict, but we know from the pipeline setup that it has the structure of FilterConfig, so this lets us access config parameters with dot notation and get autocompletion in IDEs.
        input_file = Path(ctx.require("input_file")) 
        out_dir = self._mkdir(ctx.work_dir / "filtered") # each step gets its own subdirectory under the main work_dir, which is named after the step for clarity. The _mkdir helper creates it if it doesn't exist and returns the path.
        out_file = out_dir / "library_filtered.smi" 

        # ── Resume check ──────────────────────────────────────────────────────
        # if the output file already exists, it reads the line count (subtracting 1 for header), populates
        # the context with what downstream steps need, and returns immediately.
        # this pattern is identical in every step, meaning you can restart an interrupted pipeline at any point
        # without redoing any previous work.
        if out_file.is_file():
            self.logger.info("Resuming — filtered file already exists: %s", out_file)
            n_filt = sum(1 for _ in out_file.open()) - 1  # subtract header. This is a fast way to count lines without loading the whole file into memory.
            ctx.set("filter_file", out_file)
            ctx.set("n_molecules_filtered", n_filt) 
            return ctx


        self.logger.info("Loading library from %s …", input_file)
        df = pd.read_csv(input_file, sep=" ", header=0, names=["smiles", "idnumber"]) # space-separated, two-column format with explicit column names.
        n_raw = len(df) 
        ctx.set("n_molecules_raw", n_raw)
        self.logger.info("  %d molecules loaded.", n_raw)

        # Parse SMILES -> RDKit Mol objects (invalid SMILES become None)
        PandasTools.AddMoleculeColumnToFrame(df, "smiles", "mol")
        df = df[df["mol"].notna()].copy() # filtering with .notna() before computing descriptors prevents RDKit from crashing on invalid molecules deep in a computation. The .copy() is needed to avoid SettingWithCopyWarning when we later add descriptor columns to the filtered dataframe.
        n_valid = len(df)
        if n_valid < n_raw:
            self.logger.warning(
                "  %d molecules had unparseable SMILES and were dropped.",
                n_raw - n_valid,
            )

        # ── Compute descriptors ───────────────────────────────────────────────
        # All descriptors are computed upfront in one pass. Computing descriptors is 
        # the slow step, so computing them all first also means the full data is available for any
        # future logging, export, or visualization without rerunning.
        self.logger.info("  Computing descriptors …")
        df["sLogP"]     = df["mol"].apply(Descriptors.MolLogP)
        df["RotBonds"]  = df["mol"].apply(Descriptors.NumRotatableBonds)
        df["MW"]        = df["mol"].apply(Descriptors.ExactMolWt)
        df["FSP3"]      = df["mol"].apply(rdMolDescriptors.CalcFractionCSP3)
        df["AroRings"]  = df["mol"].apply(rdMolDescriptors.CalcNumAromaticRings)
        df["AliphRings"]= df["mol"].apply(rdMolDescriptors.CalcNumAliphaticRings)
        df["TotRings"]  = df["AroRings"] + df["AliphRings"]
        df["Charge"]    = df["mol"].apply(rdmolops.GetFormalCharge)

        # ── Apply filters (each logged individually for traceability) ─────────
        # Each filter is applied by calling the _apply() helper, which applies the boolean mask, counts the 
        # remaining molecules, logs the result, and returns the filtered dataframe and new count for the
        # next filter. Threading the count through each call enables the running-total logging in apply().
        self.logger.info("  Applying filters …")
        df, n_after = self._apply(df, n_valid, "sLogP",
                                  df["sLogP"].between(cfg.slogp_min, cfg.slogp_max))
        df, n_after = self._apply(df, n_after, "RotBonds",
                                  df["RotBonds"] <= cfg.rot_bonds_max)
        df, n_after = self._apply(df, n_after, "MW",
                                  df["MW"].between(cfg.mw_min, cfg.mw_max))
        df, n_after = self._apply(df, n_after, "FSP3",
                                  df["FSP3"] >= cfg.fsp3_min)
        df, n_after = self._apply(df, n_after, "AroRings",
                                  df["AroRings"].between(cfg.aro_rings_min,
                                                         cfg.aro_rings_max))
        df, n_after = self._apply(df, n_after, "AliphRings",
                                  df["AliphRings"] <= cfg.aliph_rings_max)
        df, n_after = self._apply(df, n_after, "TotRings",
                                  df["TotRings"].between(cfg.total_rings_min,
                                                         cfg.total_rings_max))
        df, n_after = self._apply(df, n_after, "Charge",
                                  df["Charge"] == cfg.formal_charge)

        self.logger.info(
            "  Filter complete: %d / %d molecules passed (%.1f %%).",
            n_after, n_raw, 100 * n_after / max(n_raw, 1),
        )

        # ── Write output ──────────────────────────────────────────────────────
        df[["smiles", "idnumber"]].to_csv(out_file, sep=" ", index=False)
        self.logger.info("  Written to %s", out_file)

        ctx.set("filter_file", out_file)
        ctx.set("n_molecules_filtered", n_after)
        return ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply(
        df: "pd.DataFrame", 
        n_before: int,
        label: str,
        mask: "pd.Series", # boolean mask indicating which rows pass the filter condition
    ) -> tuple["pd.DataFrame", int]:
        df = df[mask] # apply the boolean mask to filter the dataframe
        n_after = len(df) # count the remaining molecules after filtering
        logger.info(
            "    %-12s → %d retained  (-%d)",
            label, n_after, n_before - n_after,
        ) # the %-12s formats the label to be left-aligned in a 12-character-wide field, which makes the log output nicely aligned and easier to read.
        return df, n_after