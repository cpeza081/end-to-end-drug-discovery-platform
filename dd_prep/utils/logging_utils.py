"""
utils/logging_utils.py — Logging configuration for the pipeline.

Sets up two handlers for the root 'dd_prep' logger:
  1. Coloured console output  (StreamHandler → stdout)
  2. Plain-text file log      (FileHandler  → work_dir/dd_prep.log)

Colour coding makes it easy to spot warnings and errors when watching
a long pipeline run in a terminal.  The file log preserves full
timestamps and is the authoritative audit record for a run.

Usage (called once at the start of Pipeline.run()):
    from dd_prep.utils.logging_utils import setup_logging
    setup_logging(work_dir=Path("./my_run"), level=logging.INFO)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ── ANSI colour codes ─────────────────────────────────────────────────────────
# Only used in the console handler; the file handler strips them automatically
# because it uses a plain Formatter rather than ColorFormatter.

_RESET  = "\033[0m"
_COLOURS = {
    logging.DEBUG:    "\033[37m",   # grey
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}


class _ColorFormatter(logging.Formatter):
    """Formatter that prepends an ANSI colour code to the level name."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, _RESET) # Get the colour code for the log level; default to no colour if the level isn't in the mapping.
        # Temporarily colour the level name; restore afterwards so other
        # handlers (e.g. the file handler) see the plain string.
        original = record.levelname 
        record.levelname = f"{colour}{record.levelname}{_RESET}" # Wrap the level name in the colour code and reset code so that only the level name is coloured in the console output.
        result = super().format(record) # Call the base class format to produce the final log message string with the coloured level name.
        record.levelname = original
        return result


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    work_dir: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure the 'dd_prep' logger.

    Safe to call multiple times. Re-calling clears any previously attached
    handlers so you don't get duplicate lines in the console.

    Parameters
    ----------
    work_dir : Optional[Path]
        If provided, a ``dd_prep.log`` file is created inside this directory.
        The directory is created if it does not exist.
    level : int
        Minimum log level.  Use ``logging.DEBUG`` for verbose step output.
    """
    root = logging.getLogger("dd_prep")
    root.setLevel(level)

    # Clear existing handlers to avoid duplicate output on re-calls.
    root.handlers.clear()

    _fmt_console = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s" # The format string for log messages in the console; includes timestamp, level, logger name, and message. The levelname is left-aligned in an 8-character field, and the name is left-aligned in a 28-character field for consistent formatting.
    _fmt_file    = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    _datefmt_console = "%H:%M:%S" # Console logs only show time for brevity, since the file log has the full date and time.
    _datefmt_file    = "%Y-%m-%d %H:%M:%S"

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout) # A stream handler is a logging handler that writes log messages to a stream; in this case, sys.stdout for console output.
    console.setLevel(level) # Set the log level for the console handler; it will only emit messages at this level or higher.
    console.setFormatter(_ColorFormatter(fmt=_fmt_console, datefmt=_datefmt_console)) 
    root.addHandler(console) # Add the console handler to the root logger so that log messages are sent to the console with the specified formatting and colour coding.

    # ── File handler ──────────────────────────────────────────────────────────
    if work_dir is not None: 
        work_dir.mkdir(parents=True, exist_ok=True) # Create the work directory if it doesn't exist; parents=True allows creating parent directories if needed, and exist_ok=True prevents an error if the directory already exists.
        log_path = work_dir / "dd_prep.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8") # Writes logging messages to a file.
        file_handler.setLevel(logging.DEBUG)   # always capture everything to file
        file_handler.setFormatter(
            logging.Formatter(fmt=_fmt_file, datefmt=_datefmt_file)
        ) # Plain formatter without colour codes for file handler.
        root.addHandler(file_handler) # Add the file handler to the root logger so that log messages are also written to the specified log file.