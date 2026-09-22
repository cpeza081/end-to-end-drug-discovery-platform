# dd_final_docking

Stage VIII (final phase) of Deep Docking: dock the survivors of the last
iteration and hand off a scored SDF.

## One execution engine, two ways to drive it

`dd_final_docking.py` supports:

- **`--campaign-config campaign.yaml`**: reads `docking.program`,
  `receptor_file`, `box_json`, `score_keyword`, etc. straight from the
  campaign, and reuses that campaign's own `dd_ligand_prep.py` (Phase 2's
  ligand prep) and `dd_autodock_export.py` (AutoDock-GPU's `.dlg` ->
  scored-SDF converter) from the sibling `dd_active_learning/` directory.
- **explicit flags** (`--backend` / `--receptor` / `--center` / `--size` /
  ...): for a one-off run, a target that was never run through
  `dd_orchestrator.py`/`campaign.yaml`, or driving each stage by hand. Uses
  this script's own RDKit/OpenEye prepare backends and a meeko-on-the-fly
  PDBQT conversion for AutoDock-GPU.

Either way, every stage after `select` reads/writes one shared,
self-describing directory (`--final-dir`) instead of single monolithic
files -- this is exactly what lets the wizard just shell out to these
subcommands instead of re-implementing anything:

```
<final-dir>/
  id_score.csv              top-N, id,score, sorted best-first
  smile/chunk_00000.smi     headerless "smiles id", up to --batch-size each
  smile/chunk_00001.smi
  ...
  sdf/chunk_00000.sdf       3D, prepared -- gnina reads these directly
  pdbqt/chunk_00000/*.pdbqt per-molecule PDBQT -- AutoDock-GPU reads these
  docked/chunk_00000_docked.sdf   one scored SDF per chunk, backend-agnostic
  ...
```

## Commands for a standalone run (you already have `id_score.csv` + `smiles.csv`)

```bash
# 1. Slice the top 100000 by DD score, chunked for the later stages
#    (re-sorts explicitly, doesn't assume the files are still in score order)
python dd_final_docking/dd_final_docking.py select \
    --smiles smiles.csv --id-score id_score.csv \
    --top-n 100000 --final-dir final_docking

# 2. Build docking-ready 3D structures for every chunk
#    (on cluster, load the same modules as dd_prep/slurm/05_omega.sh first:
#     module load gcc rdkit; export OE_LICENSE=... ; see that script)
python dd_final_docking/dd_final_docking.py prepare \
    --final-dir final_docking --backend openeye

# 3. Dock as a SLURM array job, one task per chunk
#    (edit RECEPTOR / box in the script first; set N_CHUNKS/--array to match
#    how many chunk_NNNNN.smi files "select" reported writing)
export DD_PREP_VENV=/scratch/cpeza081/dd_prep_venv
export DD_PREP_PROJECT=/home/cpeza081/end-to-end-drug-discovery-platform
sbatch --export=ALL,FINAL_DIR=final_docking,RECEPTOR=receptor.pdbqt \
    dd_final_docking/submit_dd_final_docking.slurm

# 4. Once every array task finishes, merge into the handoff file
python dd_final_docking/dd_final_docking.py merge \
    --final-dir final_docking \
    --score-tag minimizedAffinity --lower-is-better \
    --out final_top100k_docked.sdf \
    --summary-csv final_top100k_scores.csv
```

`final_top100k_docked.sdf` is the end result: one best pose per
molecule, sorted best-score-first, with the score as an SD tag on every
entry.

If you ran the campaign through `dd_orchestrator.py`, skip all of the above
and just run:

```bash
python dd_final_docking/dd_final_docking_wizard.py --config campaign.yaml
```

which does the same four steps for you, reading the receptor/box/docking
program straight from `campaign.yaml` -- see
`dd_final_docking_wizard_README.md`.

## Assumptions to double check (as of writing)

- gnina's docking box: pass either `--center X Y Z --size X Y Z` or
  `--autobox-ligand reference.sdf` to the `dock` subcommand / edit the
  SLURM script (or use `--campaign-config`, whose `docking.box_json` is
  read automatically). Neither is guessed for you since getting the box
  wrong silently docks into the wrong pocket.