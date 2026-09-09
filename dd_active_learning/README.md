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

## Ending a campaign early

If you want to stop the campaign before `total_iterations`, it is not as
simple as lowering `total_iterations` and resuming, because each iteration's hit
threshold was fixed at the moment its Phase 4a job script was generated.

Phase 4a passes two values to the DD protocol that both depend on the configured
total:

```
--total_iterations <N>    # sets the percent_first to percent_last ramp
--is_last <True|False>    # True only when iteration == total_iterations
```

Only the iteration flagged `is_last` applies `percent_last`. Every other
iteration uses an interpolated, looser cutoff. So if you configured 4 iterations
and stop after 2, iteration 2's survivors were cut at step 2 of a 4-step ramp.
That is a legitimate result, but a wider one than a campaign configured for 2
iterations would have produced.

Editing `total_iterations` now does not retroactively change what already ran.
You have two options. The first has no further compute. The set already exists on disk. Check its size first, since
that is the number that actually matters:

```bash
ITER=<project_dir>/<campaign_name>/iteration_<N>
wc -l "$ITER/morgan_1024_predictions"/* | tail -1
```

Then skip to *Cancel what is still queued* below, and extract.

The second option costs one training array. Set `total_iterations` to the iteration you want to end on, then
re-run that iteration from Phase 4:

```yaml
dd:
  total_iterations: 2
```

```bash
python dd_active_learning/dd_orchestrator.py --config campaign.yaml \
    --start-iter 2 --start-phase 4
```

Phase 4a regenerates with `is_last=True`, applies `percent_last`, and the chain
ends with the final extraction. The orchestrator rewrites job scripts on every
run, so nothing in `job_scripts/` needs hand-editing.

### Cancel what is still queued

Look before cancelling. A failed phase usually kills its own chain through
`DependencyNeverSatisfied`, so there may be nothing left.

```bash
squeue -u $USER --format="%.18i %.9T %.14r %j"
```

Then `scancel` the job IDs belonging to iterations you are abandoning.

### Extract the SMILES by hand

The orchestrator only submits `final_extraction` after the configured last
iteration, so if you choose the first of the two options then you must run it yourself. 
It takes the prediction directory as an argument and is not special to any particular iteration.

```bash
mkdir -p <project_dir>/final_iter<N> && cd <project_dir>/final_iter<N>

python "$DD_PROTOCOL_DIR/utilities/final_extraction.py" \
    -smile_dir "<library.smiles_dir from campaign.yaml>" \
    -prediction_dir "$FINAL_ITER/morgan_1024_predictions" \
    -processors 8
```

It writes `smiles.csv` and `id_score.csv` into the current working
directory, so `cd` somewhere deliberate first. Both paths must be absolute.
It parses every SMILES chunk in the prepared library, so you may want to grab
an allocation.

```bash
salloc --account=<your-account> --cpus-per-task=8 --mem=32G --time=2:00:00
```

Re-load your modules and activate the environment inside the allocation. If you
are unsure what those are, copy them from the top of any generated script in
`job_scripts/`.

The `score` column in `id_score.csv` is the **model's predicted probability of
being a virtual hit** (0-1), not a docking affinity. Only the molecules actually
docked during the campaign have `minimizedAffinity` values, and those live in
`iteration_<N>/docked/`.

### final_extraction.py requires pandas < 2

`final_extraction.py` uses the positional-axis form `df.drop('score', 1)`, which
was removed in pandas 2.0. On Alliance clusters `scipy-stack` now provides
pandas 3.x, so the script fails with:

```
TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given
```

Pin pandas to solve this issue:

```bash
pip install --no-index "pandas<2"
python -c "import pandas; print(pandas.__version__, pandas.__file__)"
```

Confirm that second line reports a 1.x version. If the virtual environment is
not writeable, pip falls back to `~/.local`, which then shadows `scipy-stack`
for every python3.11 environment you use. Undo it with `pip uninstall pandas numpy`.

---

## Known rough edges

Things the orchestrator does not handle for you, and solutions you may need to use.

### A running job's time limit can only be lowered

Only a privileged user can increase a running or suspended job's `TimeLimit`. This being the case, the size of your library and your chunks can impact the time limit needs. If you notice issues with time running out, pending jobs can be raised freely using this command.

```bash
scontrol update JobId=<id> TimeLimit=<new-limit>
```

Use the absolute form, not `TimeLimit+=`. The increment form is rejected once an
array has split into more than one job record.

### `exclude_nodes` applies to every job in a submission

For a job already queued, the field is updatable. It is `ExcNodeList`, shown below.

```bash
scontrol update JobId=<id> ExcNodeList=<excluded-nodes>
scontrol show job <id> | grep -o "ExcNodeList=[^ ]*"
```

### One faulty GPU can consume the array and job path

The Phase 3b preamble runs `nvidia-smi` before docking and exits 1 if the GPU is
unusable. The freed slot is immediately taken by the next array task, which
lands on the same bad node and dies the same way. Twelve tasks can burn in six
minutes.

The symptom is a run of tasks with elapsed times of 2-3 seconds, exit code 1,
all on one node. Add that node to `exclude_nodes` and resume. Completed shards
are skipped by the `[ -s "$OUT_FILE" ]` guard.

### Phase 3 walltime has to be measured, not guessed

Measured on Fir with 12 CPUs, one H100, and gnina at
`--cnn_scoring rescore --exhaustiveness 8 --num_modes 1`, we tested 9.7 to 19.1 hours
per shard, median around 14. The spread is node-to-node and other users'
contention, not chemistry, as shards are unsorted slices of the same sample.

`--num_modes 1` controls how many poses are written. It does not speed up docking.

To check a running array against its limit, count `$$$$` records in the
in-progress output:

```bash
grep -c '^\$\$\$\$' "$DD_ITERATION/docked_shards"/*.partial.sdf
```

Divide by that task's own elapsed time. Do not compare raw counts between tasks:
array tasks do not start together, and can begin days apart as GPUs free up.
Treat a short extrapolation as approximate, as observed error against a 3-hour
sample was up to 50%.

If you are on DRA, be aware that priority partitions are banded by walltime (3 h / 12 h / 24 h / 3 d / 7 d) and
`gpubackfill` caps at 24 hours. Staying at or under 24 hours keeps you eligible
for both the standard band and backfill. Check yours with `sinfo -o "%20P %10l"`.

### `score_keyword` must be a lower-is-better field

gnina writes several scores per pose. With `--cnn_scoring rescore` you get
`minimizedAffinity`, `CNNscore`, `CNNaffinity`, `CNNaffinity_variance` and
`CNN_VS`. Only `minimizedAffinity` is Vina-style kcal/mol where lower is better.
`CNNscore` and `CNNaffinity` are higher-is-better, and selecting one inverts
your labels **silently** - the campaign runs to completion and trains on
backwards data.

The wizard defaults to `minimizedAffinity`, so you will not hit this by leaving things alone. 
The realistic routes in are as follows.

- You choose `CNNaffinity` on purpose. gnina's own benchmarks rate its CNN
  scores above the Vina-style one, so you may choose this. 
  But `CNNaffinity` is a predicted pKd and DD
  ranks ascending. The better score, used correctly by gnina, becomes the wrong
  score the moment DD sorts on it.
- Switching docking engine and carrying the keyword over.
- Inheriting a `campaign.yaml` from a colleague or an earlier project with
  a different engine or different gnina flags.
- Changing `--cnn_scoring` to `none`. The CNN tags disappear from the
  output entirely. A keyword pointing at one of them then matches nothing, and
  label extraction produces an empty or near-empty set.

Phase 4 still happily on inverted labels and reports a respectable AUC, because the model reproduces
whatever labelling it was handed. The failure only shows up when someone
re-docks the "hits" and finds they score badly.

The way to check this is by seeing if the score is negative. A Vina-style affinity in
kcal/mol is negative for anything that binds. A pKd or a 0-1 pose score is not.

```bash
kw=$(grep score_keyword campaign.yaml | awk -F'"' '{print $2}')
f=$(ls "$ITER/docked_shards"/*_docked.sdf | head -1)
awk -v k="$kw" '$0 ~ "^> *<"k">" {getline; print $1}' "$f" | head -5
```

If those numbers come back positive, or nothing prints at all, stop the campaign
before Phase 4a runs. Also confirm the keyword exists as a tag in the output.

```bash
grep -o '^> *<[^>]*>' "$f" | sort -u
```

Check the distribution after the first docked shard. A median around
-6 to -8 kcal/mol is normal for drug-like molecules in a real pocket.

```bash
f=$(ls "$ITER/docked_shards"/*_docked.sdf | head -1)
awk '/^> *<minimizedAffinity>/{getline; print $1}' "$f" | sort -n | \
    awk '{a[NR]=$1} END{print "n="NR, "min="a[1], "median="a[int(NR/2)], "max="a[NR]}'
```

### Validation and test sets are resampled every iteration

Phase 1 redraws all three sets from the surviving library each round. 
Every iteration therefore produces three docked SDFs. If you are reasoning about sample
counts or array sizing, budget `train_size + 2 * val_size` per iteration.

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
