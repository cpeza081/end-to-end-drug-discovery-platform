"""
cli.py — Command-line interface for the DD preparation pipeline.

Registered as the ``dd-prep`` console script in pyproject.toml, so once
the package is installed you can run:

    dd-prep --config my_config.yaml
    dd-prep --input library.smi --work-dir ./prep_out --dry-run
    dd-prep --config my_config.yaml --validate-only
    dd-prep --config my_config.yaml --step filter

All arguments are optional — any value not supplied on the command line
falls back to the YAML config, which in turn falls back to the dataclass
defaults in config.py.
"""

from __future__ import annotations

import argparse # builds the CLI parser and handles parsing command-line arguments
import logging #controls log output
import sys # used for exiting with a status code
from pathlib import Path # checking file existence and constructing output paths

from dd_prep.config import load_config # loads the YAML config and applies CLI overrides
from dd_prep.pipeline import Pipeline # Main workflow runner

# Steps that operate on individual chunk files and therefore support --chunk-index.
_ARRAY_STEPS = {"flipper", "tautomer", "fingerprint", "omega"}
 
# Steps that must run once over all files and do not support --chunk-index.
_SEQUENTIAL_STEPS = {"filter", "split", "organize"}
 
_ALL_STEPS = _ARRAY_STEPS | _SEQUENTIAL_STEPS


def build_parser() -> argparse.ArgumentParser:
    """
    Construct and return the argument parser.

    Kept as a separate function so it can be imported and reused by
    tests or a GUI wrapper without executing the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="dd-prep",
        description="Deep Docking library preparation pipeline", 
        formatter_class=argparse.RawDescriptionHelpFormatter, # preserves formatting of examples
        epilog="""
examples:
  # Run with a config file (recommended)
  dd-prep --config my_config.yaml

  # Quick run without a config file, using all defaults
  dd-prep --input library.smi --work-dir ./prep_output

  # Check the config and verify binaries exist before committing to a run
  dd-prep --config my_config.yaml --validate-only

  # Preview commands without executing anything
  dd-prep --config my_config.yaml --dry-run

  # Re-run from scratch, ignoring previous checkpoint
  dd-prep --config my_config.yaml --no-resume

  # Library pre-split across several files (merged into one filtered library)
  dd-prep --input part1.smi part2.smi part3.smi --work-dir ./prep_output
  dd-prep --input 'library_parts/*.smi' --work-dir ./prep_output

  # Filter only: apply the property filters and report how many
  # molecules survive, without running any downstream step
  dd-prep --config my_config.yaml --step filter

  # Same, but force a re-filter after editing thresholds in the config
  # (without --no-resume the previous filtered file is reused)
  dd-prep --config my_config.yaml --step filter --no-resume
        """,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Path to a YAML configuration file. "
             "CLI flags override values set here.",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        nargs="+",
        help=(
            "Input SMILES library (space-separated, 'smiles idnumber' header; "
            "extra columns are ignored). "
            "Accepts several paths, or a quoted glob, for a library that is "
            "pre-split across files: -i part1.smi part2.smi  or  -i 'parts/*.smi'. "
            "Multiple files are merged into one filtered library."
        ),
    )
    parser.add_argument(
        "--work-dir", "-o",
        metavar="DIR",
        help="Root directory for all intermediate and final outputs.",
    )

    # ── Cluster / array-job flags ─────────────────────────────────────────────
    
    # '--step' allows the user to specify a single pipeline step to run, which is useful for running individual steps.
    parser.add_argument(
        "--step",
        metavar="STEP",
        choices=sorted(_ALL_STEPS),
        help=(
            "Run only this pipeline step and stop. "
            "'--step filter' applies the property filters and prints the "
            "number of molecules remaining. "
            f"Array-capable steps: {sorted(_ARRAY_STEPS)}. "
            f"Sequential steps: {sorted(_SEQUENTIAL_STEPS)}."
        ),
    )

    # '--chunk-index' allows the user to specify which chunk of the input library to process, which is useful for parallelizing across a cluster.
    parser.add_argument(
        "--chunk-index",
        type=int,
        metavar="N",
        default=None,
        help=(
            "0-based chunk file index. Used with --step for SLURM array jobs. "
            "If N >= number of chunks, the task exits cleanly (no error)."
        ),
    )

    # ── Execution flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands that would be executed without running them.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore any existing checkpoint and re-run all steps from scratch.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Run pre-flight validation checks only; do not execute the pipeline.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging (very chatty; useful for troubleshooting).",
    )

    return parser


def _print_filter_result(ctx) -> None:
    """
    Print the surviving molecule count as a single bare line on stdout.

    The logger already writes a formatted summary block, but that output
    carries timestamps and ANSI colour, which makes it awkward to scrape.
    This line is deliberately plain so a Slurm script can do:

        N=$(dd-prep -c cfg.yaml --step filter | grep '^molecules_remaining=' | cut -d= -f2)

    Does nothing if the filter step was disabled or never ran.
    """
    n = ctx.get("n_molecules_filtered") if ctx is not None else None
    if n is None:
        return
    print(f"molecules_remaining={n}")


def main(argv: list[str] | None = None) -> None:
    """
    Entry point called by the ``dd-prep`` console script.

    Parameters
    ----------
    argv : list[str] | None
        Argument list for testing. Defaults to sys.argv when None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Validate argument combinations ────────────────────────────────────────
    if args.chunk_index is not None and args.step is None:
        parser.error("--chunk-index requires --step.")
    if args.chunk_index is not None and args.step in _SEQUENTIAL_STEPS:
        parser.error(
            f"--chunk-index is not supported for sequential step '{args.step}'. "
            f"Array-capable steps are: {sorted(_ARRAY_STEPS)}."
        )

    # ── Translate CLI args into config overrides ──────────────────────────────
    overrides: dict[str, object] = {}
    if args.input:
        # argparse always hands back a list with nargs="+". Unwrap a single
        # entry so the saved run_config.yaml records the same shape the user
        # would have written by hand.
        overrides["input_file"] = args.input[0] if len(args.input) == 1 else args.input
    if args.work_dir:
        overrides["work_dir"] = args.work_dir
    if args.dry_run:
        overrides["dry_run"] = True
    if args.no_resume:
        overrides["resume"] = False

    # ── Load config (YAML + overrides layered on top of defaults) ─────────────
    cfg = load_config(yaml_path=args.config, overrides=overrides)

    # ── Validate required fields ──────────────────────────────────────────────
    if not cfg.input_file:
        parser.error(
            "input_file is required. "
            "Set it in your config file or pass --input <file> [<file> ...]."
        )
    # Resolve here so a bad path or an empty glob fails immediately with a
    # clear message, rather than part-way through a long run.
    try:
        resolved_inputs = cfg.input_files()
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not resolved_inputs:
        parser.error(f"No input files matched: {cfg.input_file!r}")

    # ── Set up logging level before pipeline initialises ─────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO # If we want more logging output, that is controlled by the --verbose flag.
    logging.basicConfig(level=level)   # root logger. pipeline will refine it

    # ── Build and run ─────────────────────────────────────────────────────────
    if len(resolved_inputs) > 1:
        print(f"Resolved {len(resolved_inputs)} input files:")
        for path in resolved_inputs:
            print(f"  {path}")

    pipeline = Pipeline(cfg)

    if args.validate_only:
        ok = pipeline.validate()
        sys.exit(0 if ok else 1) # exit with 0 if validation passed, 1 if it failed, so this can be used in scripts and CI pipelines to block execution if the config isn't valid.

    if args.step:
        # Single-step mode: used by Slurm array jobs and by filter-only runs.
        ctx = pipeline.run_single_step(
            step_name=args.step,
            chunk_index=args.chunk_index,
        )
        if args.step == "filter":
            _print_filter_result(ctx)
    else:
        # Full pipeline mode: normal local / login-node usage.
        pipeline.run()

if __name__ == "__main__":
    main() #allows cli.py to be run directly as a script, although it is normally invoked through the installed dd_prep entry point.