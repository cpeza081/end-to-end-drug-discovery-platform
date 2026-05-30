"""
cli.py — Command-line interface for the DD preparation pipeline.

Registered as the ``dd-prep`` console script in pyproject.toml, so once
the package is installed you can run:

    dd-prep --config my_config.yaml
    dd-prep --input library.smi --work-dir ./prep_out --dry-run
    dd-prep --config my_config.yaml --validate-only

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
        help="Input SMILES library (space-separated, 'smiles idnumber' header).",
    )
    parser.add_argument(
        "--work-dir", "-o",
        metavar="DIR",
        help="Root directory for all intermediate and final outputs.",
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

    # ── Translate CLI args into config overrides ──────────────────────────────
    overrides: dict[str, object] = {}
    if args.input:
        overrides["input_file"] = args.input
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
            "Set it in your config file or pass --input <file>."
        )
    if not Path(cfg.input_file).is_file():
        parser.error(f"Input file does not exist: '{cfg.input_file}'")

    # ── Set up logging level before pipeline initialises ─────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO # If we want more logging output, that is controlled by the --verbose flag.
    logging.basicConfig(level=level)   # root logger. pipeline will refine it

    # ── Build and run ─────────────────────────────────────────────────────────
    pipeline = Pipeline(cfg)

    if args.validate_only:
        ok = pipeline.validate()
        sys.exit(0 if ok else 1) # exit with 0 if validation passed, 1 if it failed, so this can be used in scripts and CI pipelines to block execution if the config isn't valid.

    pipeline.run()


if __name__ == "__main__":
    main() #allows cli.py to be run directly as a script, although it is normally invoked through the installed dd_prep entry point.