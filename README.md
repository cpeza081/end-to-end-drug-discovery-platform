# end-to-end-drug-discovery-platform
Development of an end-to-end drug discovery platform in the Gentile lab.

## Tutorial for dd_prep (deep docking preparation script)

### You will need:
 - [ ] SSH access to a cluster
 - [ ] Your SMILES library file transferred to a working directory
 - [ ] Your PI's Slurm account name
 - [ ] OpenEye toolkit access (you must have the binaries and the licence installed on the cluster and it helps to know which directory it lies in)

---

**Expected file format** — the pipeline accepts most common formats automatically, including:
 
```
SMILES ID
CCO ZINC000001
c1ccccc1 ZINC000002
```
 
or with the columns swapped, or comma-separated. The pipeline detects the format and handles it.

---

### Procedure

1. Clone repository (skip this step if the program is already on the cluster)

```bash
git clone <repo-link>
```

2. Navigate to the project:

```bash
cd ~/end-to-end-drug-discovery-platform
```

3. Run the setup wizard:

```bash
slurm/setup_cluster.sh
```

You will then be asked a series of questions. Have the answers ready.
 - Full path to the input SMILES library (.smi file)
 - Output directory (there is a default that you can have created by pressing enter)
 - Virtual environment directory (again there is a default)
 - OpenEye configuration (you may choose to either have the script find the directory or provide it yourself). If auto-detection doesn't find it, you will be asked to enter manually. If you skip OpenEye setup, Flipper and Tautomers will not run. You will be able to re run the setup wizard once you have access.

The wizard generates a 'my_run.yaml' config file with sensible defaults. You can edit this file later to adjust filter thresholds, or you can create your own 'my_run.yaml' file with your own configs and choose to overwrite it.

### Running the pipeline

```bash
cd ~/end-to-end-drug-discovery-platform
bash slurm/submit_pipeline.sh
``` 

You will see confirmation that the jobs were submitted. The pipeline will then run automatically from start to finish.

---

### Monitoring progress

**Quick status overview:**

```bash
bash slurm/status.sh
```

This shows which stages have completed, how many molecules passed each step, and any errors.

**Live-updating status** (refreshes each 10 seconds):

```bash 
watch -n 10 bash slurm/status.sh
```

Press `Ctrl+C` to stop watching.

**See all running jobs:**

```bash
squeue -u $USER
```

**Watch a specific step's log in real time:**

```bash
# Filter and split (Job 1)
tail -f logs/01_filter_split_*.log
 
# Flipper
tail -f logs/02_flipper_*_0.log
 
# Tautomers
tail -f logs/03_tautomers_*_0.log
 
# Organize and fingerprints (Job 4)
tail -f logs/04_organize_fp_*.log
```

Press `Ctrl+C` to stop tailing.

---

### Outputs

When the pipeline finishes, your output directory will contain:
 - dd_prep_output/dd_prep.log, a full log of the entire run
 - dd_prep_output/filtered/library_filtered.smi, the library after physicochemical filtering
 - dd_prep_output/smiles/... split chunks
 - dd_prep_output/library_prepared Processed SMILES strings for DD
 - dd_prep_output/library_prepared_fp Morgan fingerprints in DD format

---

## Running on a different library
 
Open your config file and change `input_file` and `work_dir`:
 
```bash
nano ~/end-to-end-drug-discovery-platform/my_run.yaml
```
 
```yaml
input_file: /scratch/yourusername/new_library.smi
work_dir:   /scratch/yourusername/new_library_output
```
 
Save and exit (`Ctrl+X`, `Y`, `Enter`), then run:
 
```bash
bash slurm/submit_pipeline.sh
```
 
The best practice is to keep one config file per library so you have a clear record of what settings were used for each run.
 
---

## Adjusting filter settings
 
Open the config file:
 
```bash
nano my_run.yaml
```
 
The filter section looks like this:
 
```yaml
filter:
  enabled: true
  slogp_min: 1.0      # Minimum lipophilicity
  slogp_max: 3.5      # Maximum lipophilicity
  rot_bonds_max: 6    # Maximum rotatable bonds
  mw_min: 300.0       # Minimum molecular weight (Da)
  mw_max: 450.0       # Maximum molecular weight (Da)
  fsp3_min: 0.25      # Minimum 3D character
```
 
To disable the filter entirely (for pre-filtered libraries):
 
```yaml
filter:
  enabled: false
```
 
---

## Cancelling a run
 
```bash
scancel -u $USER
```
 
This cancels all your running and pending jobs. To resubmit after fixing a problem, just run `bash slurm/submit_pipeline.sh` again. The pipeline will automatically skip steps that already completed.
 
---

## Troubleshooting
 
**Jobs disappeared from the queue but the pipeline did not finish**
 
The coordinator job may have failed. Check its log:
 
```bash
cat logs/coordinator_*.log
```
 
**"No module named dd_prep" error in logs**
 
The Python environment needs to be rebuilt:
 
```bash
source $SCRATCH/dd_prep_venv/bin/activate
pip install ~/end-to-end-drug-discovery-platform --force-reinstall
```
 
**"No product keys" / OpenEye licence error**
 
The OE_LICENSE path in the SLURM scripts is wrong. Re-run the setup wizard:
 
```bash
bash slurm/setup_cluster.sh
```
 
**Filter step produced 0 molecules**
 
A previous run may have left an empty output file. Delete the output directory and rerun:
 
```bash
rm -rf /scratch/yourusername/dd_prep_output
bash slurm/submit_pipeline.sh
```
 
**Merge conflict when running `git pull`**
 
```bash
git fetch origin
git reset --hard origin/Slurm-integration
```
 
---
 
## Updating the pipeline
 
When your team pushes changes to GitHub:
 
```bash
cd ~/end-to-end-drug-discovery-platform
git pull
 
# Reinstall dd_prep with the new code
source $SCRATCH/dd_prep_venv/bin/activate
pip install . --force-reinstall --quiet
```
 
---

### Short list of terms

| `PD` | Pending — job is queued and waiting for resources. |
| `R` | Running — job is currently executing. |
| `CG` | Completing — job just finished and SLURM is cleaning up. |