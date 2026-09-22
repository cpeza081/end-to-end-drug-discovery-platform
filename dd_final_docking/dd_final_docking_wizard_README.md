# dd_final_docking_wizard

Interactive setup wizard for Stage VIII (final phase) of a completed Deep
Docking campaign. It is a wrapper around `dd_final_docking.py` (the
single execution engine in this same folder). Every one of those
steps is a subprocess call into `dd_final_docking.py <subcommand>
--campaign-config campaign.yaml`, the exact same commands you'd type by hand
for a standalone run. Order of things the wizard does:
detecting the completed campaign, asking how many molecules to dock, and
generating, then submitting the Slurm job(s) that invoke the engine.

It adds the `dd_active_learning/` directory to `sys.path` at import time, so it needs
`dd_active_learning/` to stay a sibling folder of `dd_final_docking/`.

## What it does

```
python dd_final_docking_wizard.py --config campaign.yaml
```

1. **Connects to the campaign.** Reads `campaign.yaml` and
   `campaign_state.json`, finds the last iteration with a recorded phase-5b
   (inference array) job and checks that job's status via `dd_status.py`'s
   `query_job_status`. Warns and asks for confirmation if the campaign doesn't look
   finished.

2. **Ensures `final_extraction` has been run.** Looks for
   `<project_dir>/smiles.csv` and `id_score.csv`. If missing, offers to
   submit the same `final_extraction` job `dd_orchestrator.py` would (via
   `JobScriptFactory.final_extraction`), then exits so you re-run the wizard
   once it finishes.

3. **Asks how many of the top-scoring molecules to dock.** Prints the total
   candidate count from `id_score.csv` and prompts for how many to take from
   the top (skip with `--top-n`).

4. **Slices and chunks, by calling `dd_final_docking.py select`.** This
   wizard shells out to `dd_final_docking.py select --smiles ... --id-score ...
   --top-n ... --final-dir <project_dir>/final_docking --batch-size
   <batch-size>`, then counts however many `smile/chunk_NNNNN.smi` files
   that produced.

5. **Submits three jobs:**
   - **Ligand prep**: `dd_final_docking.py prepare --final-dir
     ... --campaign-config campaign.yaml`, which shells out to the existing
     `dd_ligand_prep.py` over the new chunk files.
   - **Docking array** (`--array=0-(n_chunks-1)`): one independent task per
     chunk, each running `dd_final_docking.py dock --final-dir ...
     --campaign-config campaign.yaml --array-task-id "$SLURM_ARRAY_TASK_ID"`
     -- up to `--batch-size` (default 10,000) molecules within one walltime
     budget. These are both defaults which the user can override with `--batch-size` and
     `--walltime`.
   - **Merge** (one job, skip with `--no-merge`): `dd_final_docking.py merge
     --final-dir ... --campaign-config campaign.yaml --out
     final_docking/final_top<N>_docked.sdf --summary-csv ...` collapses
     every chunk's docked SDF into one best-pose-per-molecule, score-sorted
     file plus a summary CSV. Ranks by `campaign.yaml`'s `docking.score_keyword` (`minimizedAffinity` for Gnina, `ADGPU_score` for AutoDock-GPU, both kcal/mol-style energies, so lower-is-better either way).

All three job IDs and the run's parameters (`top_n`, `batch_size`,
`n_chunks`, ...) are recorded under a `final_docking` key in
`campaign_state.json`, following the same resume-friendly pattern the rest
of the campaign uses.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--config` | required | campaign YAML |
| `--iteration` | auto-detect | treat this iteration as final instead of auto-detecting it (use if you stopped the campaign early and ran `final_extraction` by hand) |
| `--top-n` | interactive prompt | dock exactly this many top-scoring molecules |
| `--batch-size` | 10000 | molecules per array task |
| `--walltime` | this campaign's `scheduler.walltime.phase3_docking` (24:00:00 by default) | per-array-task walltime |
| `--max-concurrent` | none | throttle the array (`--array=0-N%K` syntax) |
| `--no-merge` | off | skip the merge job |
| `--yes` / `-y` | off | skip confirmation prompts (still asks --top-n unless that's also passed) |
| `--dry-run` | off | write job scripts, print what would be submitted, don't call `sbatch` |

Every prompt falls back to its default instead of crashing if stdin isn't
interactive (e.g. called from another script), so `--yes --top-n N` gives
you a fully non-interactive run.
