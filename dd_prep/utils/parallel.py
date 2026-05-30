"""
utils/parallel.py — Run a list of shell commands in parallel.

This will be used by FlipperStep, TautomerStep, and OmegaStep, all of which need to
process many chunk files independently through external OpenEye binaries.

Design notes
────────────
• ThreadPoolExecutor is used rather than ProcessPoolExecutor because the
  work is subprocess I/O, not Python CPU work.  Threads share the GIL (Global Interpreter Lock) but
  each subprocess runs in its own OS process, so there is no GIL contention.

• Commands are represented as list[str] (not raw strings) so no shell
  interpretation occurs.  This avoids quoting bugs on paths with spaces
  and is the subprocess best-practice recommendation.

• Failed commands are logged but do not abort the pool — remaining files
  continue processing.  The caller (pipeline.py) decides whether to halt.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed # For managing a pool of worker threads and collecting results as they complete.
from typing import Sequence # For type hinting a sequence of commands, where each command is a list of strings (the command and its arguments).

from tqdm import tqdm # For displaying a progress bar in the console while the commands are running.

logger = logging.getLogger("dd_prep.parallel")


def run_parallel(
    commands: Sequence[list[str]],
    n_parallel: int,
    dry_run: bool = False,
    desc: str = "Processing",
) -> list[subprocess.CompletedProcess]:
    """
    Execute *commands* using a thread pool of size *n_parallel*.

    Parameters
    ----------
    commands : Sequence[list[str]]
        Each element is one command represented as a list of tokens,
        e.g. ``["flipper", "-in", "chunk_001.smi", "-out", "chunk_001_isom.smi"]``.
    n_parallel : int
        Maximum number of commands running simultaneously.
    dry_run : bool
        If True, print the commands without executing them.
    desc : str
        Label shown in the tqdm progress bar.

    Returns
    -------
    list[subprocess.CompletedProcess]
        One result per command that actually ran (empty list in dry-run mode).
        Results are in completion order, not submission order.
    """
    if not commands:
        logger.debug("run_parallel called with empty command list — nothing to do.")
        return []

    if dry_run: # In dry-run mode, just print the commands and return an empty result list without executing anything.
        for cmd in commands:
            logger.info("[DRY RUN]  %s", " ".join(cmd)) 
        return []

    results: list[subprocess.CompletedProcess] = []
    failed: list[str] = []


    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        # Submit all jobs immediately; the pool throttles to n_parallel at once.
        future_to_cmd = {pool.submit(_run_one, cmd): cmd for cmd in commands} 

        with tqdm(total=len(commands), desc=desc, unit="file", ncols=80) as pbar:
            for future in as_completed(future_to_cmd): # Process completed futures as they finish, regardless of the order they were submitted.
                cmd = future_to_cmd[future] # Retrieve the original command corresponding to this future for logging purposes.
                cmd_str = " ".join(cmd)
                try:
                    result = future.result() # Get the result of the command execution; this will raise an exception if the command failed to run.
                    results.append(result) # Append the result to the results list; this includes the return code, stdout, and stderr of the command.
                    if result.returncode != 0:
                        logger.error(
                            "Command failed (exit %d): %s\nstderr: %s",
                            result.returncode,
                            cmd_str,
                            result.stderr.strip(),
                        )
                        failed.append(cmd_str)
                    else:
                        logger.debug("OK: %s", cmd_str)
                except Exception as exc:
                    logger.error("Command raised exception: %s — %s", cmd_str, exc)
                    failed.append(cmd_str)
                finally:
                    pbar.update(1) # Update the progress bar after each command completes, regardless of success or failure.

    if failed:
        logger.warning(
            "%d / %d command(s) failed. Check the log for details.",
            len(failed),
            len(commands),
        )

    return results # Return the list of results for the commands that were executed; this does not include any commands that were skipped due to dry-run mode. The results are in the order they completed, which may differ from the order they were submitted.


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_one(cmd: list[str]) -> subprocess.CompletedProcess:
    """
    Run a single command and capture stdout + stderr.

    Runs in a thread — safe because subprocess.run releases the GIL
    while waiting for the child process.
    """
    return subprocess.run(cmd, capture_output=True, text=True)