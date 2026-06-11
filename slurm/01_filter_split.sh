#!/bin/bash
#SBATCH --job-name=dd_filter_split
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/01_filter_split_%j.log
#SBATCH --account=def-YOURPI

# =============================================================================
# Job 1: Filter and split the raw SMILES library.
# Account and other settings are updated automatically by setup_cluster.sh
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?Set DD_PREP_CONFIG or run setup_cluster.sh}"
VENV_DIR="${DD_PREP_VENV:?Set DD_PREP_VENV or run setup_cluster.sh}"

module purge
module load StdEnv/2023
module load python/3.11
module load gcc rdkit

source "$VENV_DIR/bin/activate"

echo "========================================"
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Time:   $(date)"
echo "========================================"

dd-prep --config "$CONFIG" --step filter
dd-prep --config "$CONFIG" --step split

echo "Job 1 complete at $(date)"