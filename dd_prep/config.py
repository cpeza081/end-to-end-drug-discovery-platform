"""
config.py — Pipeline configuration.

All pipeline settings live in a YAML file.  Each pipeline stage
has its own nested section so parameters stay organised as new stages
are added.  Dataclasses are used so editors provide auto-complete and
type errors are caught early.

Loading priority (highest wins):
  1. CLI flags  (--key value)
  2. YAML file  (--config my_config.yaml)
  3. Dataclass defaults  (sensible out-of-the-box behaviour)
"""

from __future__ import annotations # for Python 3.10+ type hinting (e.g. dict[str, Any])

import copy
from dataclasses import dataclass, field, asdict # @dataclass auto-generates init, repr, etc. and asdict() converts to dict from a dataclass tree. 
# field() is used to specify default_factory for nested dataclasses, required when a default is a mutable type like a dict or list. 
# Without it, Python would share the same object instance across all instances of PipelineConfig, which is not what we want.

from pathlib import Path # for convenient path handling (e.g. Path("my_dir") / "file.txt" instead of os.path.join("my_dir", "file.txt"))
from typing import Any # for type hinting (e.g. dict[str, Any] means a dictionary with string keys and values of any type)

# Optional dependency: Pipeline works with all defaults if it is absent. Flag is checked precisely where it matters, not at import time, so error messages are contextual and only raised if you actually try to load or save a config file. 
# This way, users who don't need YAML support (e.g. those who hard-code a PipelineConfig in Python) aren't forced to install an extra package. 
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────#
#                         Per-step configuration blocks                        #
# ─────────────────────────────────────────────────────────────────────────────#

@dataclass
class FilterConfig:
    """
    RDKit-based property filter applied before any OpenEye processing.

    Mirrors the logic in the original extract_smiles.py but exposes every
    threshold as an independently adjustable parameter.  Any threshold can
    be disabled by setting it to None.

    Reference: Gentile et al. Nature Protocols 2022 — pre-filtering section.
    """
    enabled: bool = True

    # LogP window (Wildman–Crippen sLogP), targets drug-like lipophilicity.
    # These match the df.loc[(df['sLogP'] <= 3.5) & (df['sLogP'] >= 1)] line in extract_smiles.py, but feel free to adjust as needed for your library. 
    slogp_min: float = 1.0
    slogp_max: float = 3.5

    # Rotatable bond ceiling

    rot_bonds_max: int = 6

    # Exact molecular weight window (Da)
    mw_min: float = 300.0
    mw_max: float = 450.0

    # Fraction of sp³ carbons — promotes 3-D character
    # Fraction of sp^3 carbons >= 0.25 is a 3-dimensionality heuristic to avoid flat, aromatic molecules that have poor clinical success rates. This came from the original script.
    fsp3_min: float = 0.25

    # Aromatic ring count (range filter)
    aro_rings_min: int = 1
    aro_rings_max: int = 2

    # Aliphatic ring ceiling
    aliph_rings_max: int = 3

    # Total ring count (range filter)
    total_rings_min: int = 3
    total_rings_max: int = 4

    # Only keep neutral molecules. Charged species would require counter-ions that complicate docking.
    formal_charge: int = 0


@dataclass
class SplitConfig:
    """
    Split the (optionally filtered) SMILES file into equal-sized chunks.

    Chunking lets flipper / tautomers / inference run across many small
    files in parallel rather than one memory-hungry monolith.

    The DD paper recommends 10 M molecules per file; reduce on systems
    with limited RAM.
    """
    chunk_size: int = 10_000_000


@dataclass
class FlipperConfig:
    """
    OpenEye FLIPPER — enumerate unspecified stereocentres.

    Command produced:
        flipper -in <chunk>.smi -out <chunk>_isom.smi [-warts] [-enumNitrogen]

    Reference: https://docs.eyesopen.com/applications/omega/flipper.html
    """
    enabled: bool = True

    # Add a numeric suffix to each isomer name to guarantee uniqueness
    warts: bool = True

    # Enumerate pyramidal nitrogens (set True for amine-rich libraries). 
    # Defaults to False to match OpenEye's default and avoid library explosion for typical use.
    enum_nitrogen: bool = False

    # Any extra CLI flags passed verbatim to flipper. 
    # Any OpenEye flag not explicityly modelled in this config can be passed as a raw string.
    # So, behaviour can be tweaked without code changes as new OpenEye versions add features or change defaults.
    extra_args: str = ""


@dataclass
class TautomerConfig:
    """
    OpenEye TAUTOMERS — compute dominant tautomer / ionisation state at pH 7.4.

    Command produced:
        tautomers -in <chunk>_isom.smi -out <chunk>_states.smi
                  -maxtoreturn 1 [-ch3 false] [-warts false]

    -ch3 false:  avoids hybridisation changes near heteroatoms that can
                 corrupt ring systems (recommended by VS prep SOP).

    Reference: https://docs.eyesopen.com/applications/quacpac/tautomers/
    """
    enabled: bool = True
    max_to_return: int = 1
    ch3: bool = False
    warts: bool = False

    # Any extra CLI flags passed verbatim to tautomers
    extra_args: str = ""


@dataclass
class OmegaConfig:
    """
    OpenEye OMEGA — generate low-energy 3-D conformers (optional).

    This step is not required for the initial library preparation fed to the
    DD fingerprint/ML pipeline.  It is, however, required to produce dockable structures
    for the molecules sampled at each DD iteration (Stage IV of the protocol).
    Enable here only if you want the full library pre-processed in 3-D.

    mode "classic" → one conformer per molecule (recommended for large VS)
    mode "pose"    → multi-conformer oeb.gz (used with FRED docking)

    Reference: https://docs.eyesopen.com/applications/omega/
    """
    enabled: bool = False
    mode: str = "classic"       # "classic" | "pose"
    max_confs: int = 1
    strict_stereo: bool = False
    mpi_np: int = 8
    output_format: str = "sdf.gz"  # "sdf.gz" for Glide, "oeb.gz" for FRED

    # Any extra CLI flags passed verbatim to oeomega
    extra_args: str = ""


@dataclass
class FingerprintConfig:
    """
    RDKit Morgan fingerprint calculation — produces the feature vectors
    consumed by the DD deep-learning model.

    DD stores fingerprints as comma-separated indices of set bits rather
    than the full binary vector (saves ~10× disk space on sparse FPs).

    Format per molecule:
        <mol_name>,<bit_idx_1>,<bit_idx_2>,...

    This is the format expected by DD's sampling.py and
    extracting_morgan.py scripts.

    Reference: Gentile et al. Nature Protocols 2022, Box 1.
    """
    enabled: bool = True
    radius: int = 2       # Morgan radius — DD uses radius 2, mandated by the protocol, as cited in docstring.
    n_bits: int = 1024    # Fingerprint length — DD uses 1024 bits
    n_workers: int = 8    # Parallel worker processes


# ───────────────────────────────────────────────────────────────────────────── #
#                      Top-level pipeline configuration                         #
# ───────────────────────────────────────────────────────────────────────────── #

@dataclass
class PipelineConfig:
    """
    Master configuration object for the full preparation pipeline.

    The recommended way to supply settings is via a YAML file:

        python run_prep.py --config my_config.yaml

    Every nested section corresponds to one pipeline stage and can be
    omitted in the YAML if defaults are acceptable.
    """

    # ── I/O ──────────────────────────────────────────────────────────────
    # Path to the raw SMILES library.  Expected format: two-column
    # space-separated file with headers 'smiles' and 'idnumber'.
    input_file: str = ""

    # Root working directory; all intermediate and final outputs live here.
    work_dir: str = "./dd_prep_workdir"

    # ── Execution ────────────────────────────────────────────────────────────
    # Maximum number of chunk files processed simultaneously for
    # OpenEye steps (flipper, tautomers, omega). 
    # It is a resource constraint that applies across all external tool steps.
    n_parallel: int = 4

    # Resume a previously interrupted run: skip steps whose output
    # already exists on disk. 
    # If the pipeline crashes halfway through, this lets you fix the issue and pick up where you left off without re-running everything.
    resume: bool = True

    # Show what would be executed without actually running it.
    # Helps to clarify the effects of a config on a new cluster before submitting a long job.
    dry_run: bool = False

    # ── Stage configs ─────────────────────────────────────────────────────────
    filter: FilterConfig = field(default_factory=FilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    flipper: FlipperConfig = field(default_factory=FlipperConfig)
    tautomer: TautomerConfig = field(default_factory=TautomerConfig)
    omega: OmegaConfig = field(default_factory=OmegaConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)


# ─────────────────────────────────────────────────────────────────────────────#
#                      Loader helpers
# ─────────────────────────────────────────────────────────────────────────────#

def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursive dictionary merge: *overrides* into *base* (in place), return base.
    When both base and overrides have a dict at a given key, merge them rather than replacing the whole dict.
    This allows partial overrides of nested sections without needing to specify every parameter."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_config(d: dict) -> PipelineConfig:
    """Convert a plain dict (from YAML) into a PipelineConfig."""
    cfg = PipelineConfig()
    sub_classes = {
        "filter": FilterConfig,
        "split": SplitConfig,
        "flipper": FlipperConfig,
        "tautomer": TautomerConfig,
        "omega": OmegaConfig,
        "fingerprint": FingerprintConfig,
    }
    for key, value in d.items():
        if key in sub_classes and isinstance(value, dict):
            sub_cfg = sub_classes[key](**{
                k: v for k, v in value.items()
                if k in sub_classes[key].__dataclass_fields__
            })
            setattr(cfg, key, sub_cfg)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_config(yaml_path: str | None = None, overrides: dict | None = None) -> PipelineConfig:
    """
    Load a PipelineConfig from an optional YAML file, then apply any
    programmatic overrides supplied as a flat dict (e.g. from CLI flags).

    Parameters
    ----------
    yaml_path : str | None
        Path to a YAML configuration file.  If None, defaults are used.
    overrides : dict | None
        Key-value pairs that take precedence over everything else.
        Nested keys use dot notation: ``{"filter.slogp_max": 4.0}``.

    Returns
    -------
    PipelineConfig
    """
    if yaml_path is not None and not _YAML_AVAILABLE: # Only raise an error if the user actually tries to load a YAML config, not at import time, so users who hard-code a PipelineConfig in Python aren't forced to install PyYAML.
        raise ImportError(
            "PyYAML is required to load a config file: pip install pyyaml"
        )

    raw: dict[str, Any] = asdict(PipelineConfig())  # start with all defaults, layer changes on top.

    if yaml_path is not None: 
        with open(yaml_path) as fh: # Load the YAML file into a dict. If the file is empty, yaml.safe_load returns None, so we use "or {}" to ensure we get an empty dict instead.
            user_yaml = yaml.safe_load(fh) or {}
        _deep_update(raw, user_yaml) # Merge the user YAML on top of the defaults, so that any keys specified in the YAML override the defaults, while unspecified keys remain unchanged.

    # Dot-notation override resolution. 
    # The loop walks down the nested dict for all parts except the last, creating intermediate dicts as needed, 
    # then sets the final part to the override value. 
    # This allows users to override deeply nested parameters without needing to specify the entire section in YAML.
    if overrides: 
        for dotted_key, value in overrides.items(): #loop through each override pair.
            parts = dotted_key.split(".") #split keys at each dot. a
            target = raw
            for part in parts[:-1]: #Loop through all parts except the last.

                # setdefault() checks if part already exists as a key in the current level of the dict. 
                # if it does, it returns the existing value (which should be a dict). 
                # if it doesn't, it creates a new empty dict at that key and returns it. 
                # This way, we ensure that the intermediate levels of the nested dict exist as we walk down to the final key.
                # Typos or logical errors in the dotted keys will create new nested dicts rather than overwriting existing config sections, which provides some safety against accidental overrides.
                target = target.setdefault(part, {}) 
            target[parts[-1]] = value # Finally, set the last part to the override value.

    return _dict_to_config(raw)


def save_config(cfg: PipelineConfig, path: str) -> None:
    """Serialises the entire config back to YAML. Uses asdict, which flattens the dataclass tree 
    into a plain dict that YAML can handle. This produces the audit trail file that should be saved 
    alongside every pipeline run for reproducibility."""
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "w") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False, sort_keys=False)