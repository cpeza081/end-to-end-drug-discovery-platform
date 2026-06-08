#!/bin/bash
#SBATCH --job-name=dd_filter_split
#SBATCH --time=04:00:00          # Adjust: ~1h per 100M molecules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # Pandas filter uses multiple cores
#SBATCH --mem=32G                # Adjust down for smaller libraries
#SBATCH --output=logs/01_filter_split_%j.log
#SBATCH --account=rrg-checco89    # ← CHANGE THIS to your PI's account

# =============================================================================
# Job 1: Filter and split the raw SMILES library.
#
# Runs the filter step (RDKit property filters) and the split step (chunking
# the library into fixed-size files).  Both are sequential and single-node.
#
# Output:
#   work_dir/filtered/library_filtered.smi
#   work_dir/smiles/smiles_all_001.smi  …  smiles_all_NNN.smi
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?Set DD_PREP_CONFIG or pass via --export to sbatch}"
VENV_DIR="${DD_PREP_VENV:?Set DD_PREP_VENV or pass via --export to sbatch}"

# ── Load modules (must match 00_setup_env.sh) ─────────────────────────────────
module purge
module load StdEnv/2023
module load python/3.11

source "$VENV_DIR/bin/activate"

echo "========================================"
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Time:   $(date)"
echo "========================================"

# Run only the filter and split steps; all other steps are handled by later
# jobs. --no-resume ensures a clean run if resubmitting.
dd-prep --config "$CONFIG" --step filter
dd-prep --config "$CONFIG" --step split

echo "Job 1 complete at $(date)"