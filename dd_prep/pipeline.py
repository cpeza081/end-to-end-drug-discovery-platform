"""
pipeline.py — Pipeline orchestrator.

The Pipeline class is the single entry point for running the preparation
workflow.  It owns the step ordering, the checkpoint system, and the
top-level logging setup.

Checkpoint system
─────────────────
Completed step names are written to ``work_dir/.checkpoint.json`` after
each step succeeds.  On a resumed run, any step whose name already appears
in that file is skipped.  Deleting ``.checkpoint.json`` forces a full
re-run from scratch (equivalent to ``--no-resume``).
─────────────────

Adding a new step
─────────────────
1. Create ``dd_prep/steps/my_step.py`` (subclass PipelineStep).
2. Add its config dataclass to ``config.py``.
3. Add one line to ``_build_steps()`` below — that is the only place
   in the entire codebase that must change. 
─────────────────
NOTE: pipeline.py includes imports from all steps, including ones that are not built yet as of this writing. 
This program will therefore produce an error in its current form as of May 29, 2026, 6:42PM EST.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from dd_prep.config import PipelineConfig, save_config
from dd_prep.steps.base import PipelineContext, PipelineStep
from dd_prep.steps.filter_step import FilterStep
from dd_prep.steps.fingerprint_step import FingerprintStep
from dd_prep.steps.flipper_step import FlipperStep
from dd_prep.steps.omega_step import OmegaStep
from dd_prep.steps.organize_step import OrganizeStep
from dd_prep.steps.split_step import SplitStep
from dd_prep.steps.tautomer_step import TautomerStep
from dd_prep.utils.logging_utils import setup_logging

logger = logging.getLogger("dd_prep.pipeline")


class Pipeline:
    """
    Orchestrates the full DD library preparation pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Fully populated configuration object (from load_config() or
        constructed directly).

    Examples
    --------
    Minimal programmatic usage::

        from dd_prep.config import load_config
        from dd_prep.pipeline import Pipeline

        cfg = load_config("my_config.yaml")
        Pipeline(cfg).run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.work_dir = Path(config.work_dir)
        self._checkpoint_path = self.work_dir / ".checkpoint.json"
        self.steps: list[PipelineStep] = self._build_steps()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """
        Run pre-flight checks on all enabled steps.

        Collects every error from every step before reporting so you see
        all problems at once rather than fixing them one at a time.

        Returns
        -------
        bool
            True if all checks passed and the pipeline is safe to run.
        """
        ctx = self._build_context()
        all_errors: list[str] = []

        for step in self.steps:
            if not step.enabled:
                continue
            errors = step.validate(ctx)
            for err in errors:
                logger.error("  [%s] %s", step.name, err)
            all_errors.extend(errors)

        if all_errors:
            logger.error(
                "%d validation error(s). Resolve them before running the pipeline.",
                len(all_errors),
            )
            return False

        logger.info("All pre-flight checks passed.")
        return True

    def run(self) -> PipelineContext:
        """
        Execute the full pipeline, respecting ``resume`` and ``dry_run`` flags.

        Returns the final PipelineContext so callers can inspect outputs
        (e.g. for integration into a larger platform).
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.work_dir)

        # Snapshot the exact config used for this run — essential for
        # reproducibility and debugging in a long-running platform.
        try:
            save_config(self.config, str(self.work_dir / "run_config.yaml"))
        except Exception as exc:
            logger.warning("Could not save run config snapshot: %s", exc)

        ctx = self._build_context()
        checkpoint = self._load_checkpoint()

        self._banner()

        total_start = time.time()

        for step in self.steps:
            # ── Skip disabled steps ───────────────────────────────────────────
            if not step.enabled:
                logger.info("  [%-12s]  SKIPPED  (disabled in config)", step.name)
                continue

            # ── Skip already-completed steps on resume ────────────────────────
            if self.config.resume and checkpoint.get(step.name) == "done":
                logger.info("  [%-12s]  SKIPPED  (already completed)", step.name)
                # We still need to re-populate the context for subsequent steps.
                ctx = self._repopulate_context(step, ctx)
                continue

            # ── Run the step ──────────────────────────────────────────────────
            logger.info("  [%-12s]  START    %s", step.name, step.description)
            t0 = time.time()

            ctx = step.run(ctx)

            elapsed = time.time() - t0
            logger.info("  [%-12s]  DONE     %.1f s", step.name, elapsed)

            # Persist checkpoint so a crash after this step doesn't redo it.
            checkpoint[step.name] = "done"
            self._save_checkpoint(checkpoint)

        total_elapsed = time.time() - total_start
        logger.info("=" * 62)
        logger.info("Pipeline complete in %.1f s.", total_elapsed)
        logger.info("Outputs: %s", self.work_dir.resolve())
        logger.info("=" * 62)

        return ctx

    # ── Step registry ─────────────────────────────────────────────────────────

    def _build_steps(self) -> list[PipelineStep]:
        """
        Ordered list of all pipeline steps.

        To add a new step: import it above, then add one line here.
        The order here is the execution order.
        """
        cfg = self.config
        return [
            FilterStep(cfg.filter),
            SplitStep(cfg.split),
            FlipperStep(cfg.flipper),
            TautomerStep(cfg.tautomer),
            OrganizeStep(cfg.organize),
            FingerprintStep(cfg.fingerprint),
            OmegaStep(cfg.omega),
        ]

    # ── Context management ────────────────────────────────────────────────────

    def _build_context(self) -> PipelineContext:
        """Initialise a fresh PipelineContext from the current config."""
        ctx = PipelineContext(work_dir=self.work_dir)
        ctx.set("input_file", self.config.input_file)
        ctx.set("n_parallel", self.config.n_parallel)
        ctx.set("dry_run",    self.config.dry_run)
        ctx.set("resume",     self.config.resume)
        return ctx

    def _repopulate_context(
        self, step: PipelineStep, ctx: PipelineContext
    ) -> PipelineContext:
        """
        When a step is skipped on resume, scan its expected output directory
        and re-populate the context keys that it would have set.

        Without this, a step skipped on resume leaves holes in the context
        that break all later steps.
        """
        smiles_dir   = self.work_dir / "smiles"
        lib_dir      = self.work_dir / "library_prepared"
        fp_dir       = self.work_dir / "library_prepared_fp"
        sdf_dir      = self.work_dir / "sdf"
        oeb_dir      = self.work_dir / "oeb"

        if step.name == "filter":
            f = self.work_dir / "filtered" / "library_filtered.smi"
            if f.is_file():
                ctx.set("filter_file", f)

        elif step.name == "split":
            files = sorted(smiles_dir.glob("smiles_all_*.smi")) if smiles_dir.exists() else []
            ctx.set("chunk_files", files)

        elif step.name == "flipper":
            files = sorted(smiles_dir.glob("*_isom.smi")) if smiles_dir.exists() else []
            ctx.set("isom_files", files)

        elif step.name == "tautomer":
            files = sorted(smiles_dir.glob("*_states.smi")) if smiles_dir.exists() else []
            ctx.set("state_files", files)

        elif step.name == "organize":
            files = sorted(lib_dir.glob("*.txt")) if lib_dir.exists() else []
            ctx.set("prepared_files", files)

        elif step.name == "fingerprint":
            files = sorted(fp_dir.glob("*.txt")) if fp_dir.exists() else []
            ctx.set("fp_files", files)

        elif step.name == "omega":
            for d in (sdf_dir, oeb_dir):
                if d.exists():
                    ctx.set("sdf_files", sorted(d.iterdir()))
                    break

        return ctx

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict:
        if self._checkpoint_path.is_file():
            with open(self._checkpoint_path) as fh:
                return json.load(fh)
        return {}

    def _save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["_last_updated"] = datetime.now().isoformat()
        with open(self._checkpoint_path, "w") as fh:
            json.dump(checkpoint, fh, indent=2)

    # ── Display ───────────────────────────────────────────────────────────────

    def _banner(self) -> None:
        enabled  = [s.name for s in self.steps if s.enabled]
        disabled = [s.name for s in self.steps if not s.enabled]
        logger.info("=" * 62)
        logger.info("Deep Docking library preparation pipeline")
        logger.info("  Input  : %s", self.config.input_file)
        logger.info("  Output : %s", self.work_dir.resolve())
        logger.info("  Steps  : %s", ", ".join(enabled))
        if disabled:
            logger.info("  Off    : %s", ", ".join(disabled))
        logger.info("  Resume : %s   Dry-run : %s",
                    self.config.resume, self.config.dry_run)
        logger.info("=" * 62)