# Deep Docking active-learning campaign automation tutorial

Everything after `dd_prep` finishes: turning a prepared library plus a receptor
into a ranked set of virtual hits.

`dd_prep` gets you a library. This module screens it. The two are separate
because library preparation is done once per library, while a campaign is run
once per target.

---

One iteration is five phases:

| Phase | Name | Where | What |
|-------|------|-------|------|
| 1 | Sampling | CPU | Draw train/validation/test sets from the surviving library |
| 2 | Ligand prep | CPU | SMILES → 3-D, protonated, docking-ready (RDKit + Meeko) |
| 3 | Docking | GPU | Dock the sample with gnina or AutoDock-GPU |
| 4 | Training | GPU | Train `num_models` networks on fingerprints → docking score |
| 5 | Inference | GPU | Score every remaining molecule; keep the predicted top fraction |

After the last iteration, a final extraction writes the SMILES of the surviving
virtual hits, ready for careful re-docking.

---

## Before you start

You need all four of these:

1. **A finished `dd_prep` run** producing `library_prepared/` and
   `library_prepared_fp/`. You may check using a command like such:

   ```bash
   W=/path/to/your/dd_prep_output
   ls "$W/library_prepared" | wc -l       
   ls "$W/library_prepared_fp" | wc -l    
   ```

   The two counts must be equal. Phase 5 pairs them by chunk index, so a
   missing fingerprint file means an unscored slice of your library.

2. **A receptor** as PDB, with waters and ligands stripped unless you
   deliberately want them.

3. **A binding site.** Either a reference ligand in the pocket, or the six box
   numbers if you already know them.

4. **A docking engine on the cluster** — gnina or AutoDock-GPU, plus the module
   names that provide them.

5. Your scheduler details:

```yaml
scheduler:
  account: "rrg-yourpi"
  gpu_type: "h100"      # Nibi/Fir/Rorqual -> h100, Narval -> a100, Beluga/Graham -> v100
  array_throttle: 10    # concurrent array tasks; each uses one GPU
```

`gpu_type` is required on Alliance clusters. Check yours with
`sinfo -o "%G"` if unsure.

---

## Set up the environment

```bash
bash dd_active_learning/setup_active_learning.sh
```

The wizard finds or clones the DD protocol scripts, picks the docking engine,
records the modules that provide it, builds the conda/pip environment, collects
the DD parameters, and writes a `campaign.yaml`.

The environment must provide `rdkit`, `meeko`, `tensorflow`, `pandas`, `numpy`.
If you would rather build it by hand, use `dd_environment.yml` (conda) or
`dd_requirements.txt` (pip), then point `env.conda_env` at it.

---

## Optional: Point the campaign at your library

This would be done in step 1 if you already have the directory of your finished run, but you can do it by hand here as well.

```bash
python dd_active_learning/dd_link.py \
    --prep-work-dir /path/to/your/dd_prep_output \
    --campaign campaign.yaml
```

This sets `library.smiles_dir` and `library.fingerprint_dir`, and confirms the
prep run finished. Use `--check-only` to inspect without writing.

---

## Prepare the receptor and box

When running setup_active_learning.sh, choose how the binding
site is defined and point your config file (campaign.yaml) at it.

**If you have a co-crystallised ligand:**

```yaml
docking:
  receptor_file: "$SCRATCH/receptor/receptor.pdb"
  box_json:      "$SCRATCH/receptor/receptor_box.json"
  site:
    method: "reference_ligand"
    reference_ligand: "$SCRATCH/receptor/ref_ligand.sdf"
    padding: 4.0
```

**If you already know the box:**

```yaml
docking:
  site:
    method: "manual"
    center: [-7.28, 11.728, 3.238]
    size:   [21.75, 21.75, 23.25]
```

**If you have neither**, `method: "p2rank"` predicts the pocket from the
protein alone.

You choose these options and configure them within the script, though you can also later do it manually like this:

```bash
python dd_active_learning/dd_receptor_prep.py --config campaign.yaml
```

This writes `receptor_box.json` (and, for AutoDock-GPU, the map files). Run it
once; every iteration reuses the result.

A reference ligand gives a box that hugs the co-crystallised pose. That is
usually right, but not always. If the site is a tunnel or groove where you
expect ligands to extend beyond the reference, a manual box is the better
choice.

---

## Set the DD parameters

The setup script will ask for these parameters, which you may update in your campaign file.

```yaml
dd:
  total_iterations: 4       # paper recommends 4-11
  train_size: 1000000       # molecules sampled for training each iteration
  val_size: 1000000         # validation AND test sets, fixed at iteration 1
  percent_first: 1.0        # top 1% counts as a "virtual hit" in iteration 1
  percent_last: 0.01        # top 0.01% in the final iteration
  recall: 0.90              # fraction of true hits the model must retrieve
  num_models: 24            # models trained per iteration (16/24/48/72/144)
```

- **`total_iterations`** is the main cost lever. Each iteration docks roughly
  `train_size + val_size` molecules. Start at 4. Extend later if the enrichment
  justifies it. The orchestrator resumes, so extending is cheap.
- **`percent_first` → `percent_last`** is the tightening schedule. The library
  shrinks toward the top fraction as iterations progress.
- **`recall`** trades library size against risk. At 0.90 the model keeps enough
  molecules to retain 90% of true hits. Lower is more aggressive and cheaper;
  it also throws away more real binders.
- **`num_models`** is an ensemble over hyperparameters. More models means a
  better pick and a longer Phase 4.

---

## Validate before spending anything

The setup script will automatically validate, but you may choose to do so yourself if you manually change parameters after running the setup script.

```bash
python dd_active_learning/dd_validate.py --config campaign.yaml
```

Checks paths, tools, and environment. Fix
everything it reports.

Optionally, find bad GPU nodes first and exclude them:

```bash
bash dd_active_learning/dd_gpu_probe.sh
```

Feed the result into `scheduler.exclude_nodes`.

---

## Dry run

```bash
python dd_active_learning/dd_orchestrator.py --config campaign.yaml --dry-run
```

Prints every job script without submitting. Read them. Confirm the paths,
account, modules, and box are what you expect. This is the cheapest place to
catch a mistake.

---

## Check walltimes and resource allocation:

Look within the campaign.yaml file for your specific run to see the resource allocation defaults. Change them as necessary with what knowledge of your specific cluster that you have. 

```yaml
resources:
    phase1_sampling:  {nodes: 1, cpus: 60,  mem: "48G",  gpus: 0}
    phase2_ligand_prep: {nodes: 3, cpus: 60, mem: "32G",  gpus: 0}
    phase3_docking:   {nodes: 1, cpus: 6,   mem: "48G",  gpus: 1}   # GPU: Gnina / AutoDock-GPU
    phase4a_labels:   {nodes: 1, cpus: 8,   mem: "16G",  gpus: 0}   # CPU: labels + script gen
    phase4_training:  {nodes: 1, cpus: 6,   mem: "32G",  gpus: 1}   # GPU: per model (array task)
    phase4c_eval:     {nodes: 1, cpus: 8,   mem: "32G",  gpus: 0}   # CPU: evaluate + pick best
    phase5a_predgen:  {nodes: 1, cpus: 2,   mem: "8G",   gpus: 0}   # CPU: writes per-chunk scripts
    phase5_inference: {nodes: 1, cpus: 6,   mem: "32G",  gpus: 1}   # per array task (one chunk)
    final_extraction: {nodes: 1, cpus: 60,  mem: "32G",  gpus: 0}
```


---

## Launch

```bash
python dd_active_learning/dd_orchestrator.py --config path/to/campaign.yaml
```

The orchestrator submits one job per phase with scheduler dependencies, so the
whole campaign runs unattended. Job IDs are recorded in
`<project_dir>/campaign_state.json`.

Monitor:

```bash
python dd_active_learning/dd_status.py --config path/to/campaign.yaml
squeue -u $USER
```

After running, you may encounter an issue where resources are not being properly allocated to you. In this case, you may change your walltime to increase priority.

```bash
scontrol update JobId=<job-id> TimeLimit=<time-limit>
```

---

## When something fails

Long campaigns may sometimes fail due to outages or other issues. The
orchestrator is built such that `campaign_state.json` records what completed.

```bash
# resume from the last completed phase
python dd_active_learning/dd_orchestrator.py --config campaign.yaml --resume

# or restart from a specific point after fixing something by hand
python dd_active_learning/dd_orchestrator.py --config campaign.yaml \
    --start-iter 3 --start-phase 4
```

Diagnosing:

```bash
sacct -u $USER --starttime today \
      --format=JobID%15,JobName%22,State,ExitCode,Elapsed,MaxRSS
```

- `ExitCode 127` — a command wasn't found. Modules or environment didn't load.
- `State OUT_OF_MEMORY` — raise `mem` for that phase in `scheduler.resources`.
- `State TIMEOUT` — raise the phase's entry in `scheduler.walltime`.
- Array task failed on one node — likely a bad GPU. Add it to `exclude_nodes`
  and resume.

Raise walltime and memory for the *phase* that failed, not globally. Phase 3
and Phase 5 are the long ones.

---

## Results

After the final iteration:

```
<project_dir>/final/          SMILES of the surviving virtual hits
<project_dir>/iteration_N/    per-iteration models, scores, and logs
<project_dir>/campaign_state.json
```

The final SMILES are candidates. They are the molecules a
fingerprint model predicts would score well. The standard next step is to
dock them properly with full sampling, then inspect poses by hand before
ordering anything.

---

## Cost estimate

For a 40M-molecule library, 4 iterations, `train_size` and `val_size` at 1M:

| Phase | Work | Rough cost |
|-------|------|-----------|
| 2 Ligand prep | ~2M ligands/iteration | CPU-heavy |
| 3 Docking | ~2M ligands/iteration | 1–3 s/ligand/GPU |
| 4 Training | `num_models` networks | Minutes to hours per model |
| 5 Inference | Whole surviving library | Fast per molecule, but the library is large |

Phase 3 dominates. `array_throttle` and your GPU allocation set the wall-clock.

Worth doing once before committing: run a single iteration with
`total_iterations: 1` and a small `train_size` (say 50,000) to time Phase 3 on
your actual hardware. Multiply out from there.

---

## Reference

| Command | Purpose |
|---------|---------|
| `setup_active_learning.sh` | Interactive setup, writes `campaign.yaml` |
| `dd_link.py` | Wire a `dd_prep` output into a campaign |
| `dd_receptor_prep.py` | Build the box and receptor files |
| `dd_validate.py` | Pre-flight checks |
| `dd_gpu_probe.sh` | Find faulty GPU nodes |
| `dd_orchestrator.py` | Run the campaign |
| `dd_status.py` | Progress dashboard |
| `dd_autodock_export.py` | Convert AutoDock-GPU results for Phase 4 |

Based on Gentile et al., *Nature Protocols* 2022 (Deep Docking).
