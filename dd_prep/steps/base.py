"""
steps/base.py — Abstract base class for every pipeline step.

All pipeline stages inherit from PipelineStep and receive a PipelineContext
that carries the shared state (paths, molecule counts, …) between steps.
Adding a new step is as simple as:

    1. Create a new file in dd_prep/steps/
    2. Subclass PipelineStep, set ``name``, implement ``run()``
    3. Register it in pipeline.py's _build_steps()
"""

from __future__ import annotations # for Python 3.10+ type hinting (e.g. PipelineContext in PipelineStep.run())

import logging # for logging in steps; configured by the main pipeline runner
from abc import ABC, abstractmethod # for defining the abstract PipelineStep base class
from dataclasses import dataclass, field # for the PipelineContext dataclass, which carries shared state between steps
from pathlib import Path # for type hinting and path manipulations in PipelineContext and steps
from typing import Any # for type hinting of the flexible PipelineContext._data dictionary


# ─────────────────────────────────────────────────────────────────────────────
# Shared pipeline state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineContext:
    """
    Lightweight carrier object passed into – and returned from – every step.

    Mandatory fields are populated by the pipeline before the first step runs.
    Steps communicate with later steps by calling ``ctx.set(key, value)``.

    Conventions
    -----------
    "input_file"          – Path to the raw SMILES library
    "work_dir"            – Root working directory (Path)
    "chunk_files"         – List[Path] of split SMILES chunks (post-split)
    "isom_files"          – List[Path] of isomer-expanded chunks (post-flipper)
    "state_files"         – List[Path] of protonation-state chunks (post-tautomer)
    "prepared_files"      – List[Path] inside library_prepared/ (post-organize)
    "fp_files"            – List[Path] inside library_prepared_fp/ (post-fingerprint)
    "sdf_files"           – List[Path] of 3-D SDF files (post-omega, optional)
    "n_molecules_*"       – int counts at each stage (for progress reporting)
    """
    work_dir: Path # First-class attribute because every step needs it and it never changes; the rest of the state is flexible and stored in _data
    _data: dict[str, Any] = field(default_factory=dict, repr=False) # Flexible key-value store for arbitrary state; steps can set/get any keys they want

    #── Flexible key-value store interface ───────────────────────────────────────
    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # get() is for optional context with a default if absent, require() is for mandatory inputs where absence means broken pipeline and it fails immediately with a message.
    def require(self, key: str) -> Any:
        """Return value or raise KeyError with a helpful message."""
        if key not in self._data:
            raise KeyError(
                f"Pipeline context is missing required key '{key}'. "
                "Ensure the preceding step ran successfully."
            )
        return self._data[key]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract pipeline step
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStep(ABC):
    """
    Base class for all preparation pipeline stages.

    Subclasses must set ``name`` (used for logging and checkpoint keys)
    and implement ``run()``.  ``validate()`` is optional but strongly
    encouraged — it lets the pipeline fail fast before any expensive
    processing begins.
    """

    # ────class-level attributes──────────────────────────────────────────────────────
    # pipeline can inspect thse without instantiating the step, useful for dry-run and step registration.

    #: Human-readable identifier; must be unique across all steps.
    name: str = "base_step"

    #: One-line description shown in dry-run and progress output.
    description: str = ""
    #────────────────────────────────────────────────────────────────────────────────────

    def __init__(self, config: Any) -> None:
        """
        Each step gets its own logger namespaced under dd_prep, telling you which stage produced a message.
        Therefore, log levels can be controlled per-stage without affecting others.

        Parameters
        ----------
        config :
            The step-specific config section from PipelineConfig
            (e.g. FilterConfig, FlipperConfig, …).
        """
        self.config = config
        self.logger = logging.getLogger(f"dd_prep.{self.name}")

    # ── Core interface ────────────────────────────────────────────────────────

    # here we declare an abstract method run() from Python's abc module, so PipelineStep cannot be instantiated directly.
    # it is an abstract base class, meant to be subclassed by concrete steps that implement run(). Any subclass 
    # that doesn't implement run() will also be abstract and cannot be instantiated, raising a TypeError at runtime.
    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute the step.

        Must read its inputs from *ctx*, perform its work, update *ctx*
        with its outputs, and return the (modified) context.
        """

    def validate(self, ctx: PipelineContext) -> list[str]:
        """
        Pre-flight validation.  Return a list of human-readable error
        strings.  An empty list means all checks passed.

        Called by the pipeline before ``run()`` is invoked.  Override
        to add step-specific checks (e.g. verify an OpenEye binary exists).
        """
        return []

    # ── Convenience helpers available to all steps ────────────────────────────

    @property
    def enabled(self) -> bool:
        """
        Whether the step should be executed.  Reads ``config.enabled`` if
        present; returns True otherwise (steps are on by default).
        """
        return getattr(self.config, "enabled", True) # if a config class has no 'enabled' attribute, default to True. This makes enabled optional for steps like organise, which don't need a disable.

    def _mkdir(self, path: Path) -> Path:
        """Create *path* (and parents) if it does not exist; return it."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_binary(self, binary: str) -> list[str]:
        """
        Return an error list if *binary* is not on PATH, empty list if found.
        Useful in validate() for steps that shell out to external tools.
        """
        import shutil # we import here to avoid adding it as a global dependency since only some steps need it
        if shutil.which(binary) is None:
            return [
                f"Required binary '{binary}' not found on PATH. "
                f"Ensure it is installed and the licence is available."
            ]
        return []