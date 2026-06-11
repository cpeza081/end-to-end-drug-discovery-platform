#!/bin/bash
#SBATCH --job-name=dd_organize_fp
#SBATCH --time=06:00:00          # Adjust: ~1h per 100M molecules for fingerprints
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Fingerprint step uses multiprocessing — give it cores
#SBATCH --mem=32G
#SBATCH --output=logs/04_organize_fp_%j.log
#SBATCH --account=rrg-checco89    # ← CHANGE THIS

# =============================================================================
# Job 4: Organize into library_prepared/ and compute Morgan fingerprints.
#
# Organize is sequential (file copies).  Fingerprint processing is parallel
# within this single job — set fingerprint.n_workers in your config to match
# the --cpus-per-task value above.
#
# Input:  work_dir/smiles/smiles_all_NNN_states.smi
# Output: work_dir/library_prepared/smiles_all_NNN.txt
#         work_dir/library_prepared_fp/smiles_all_NNN.txt
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?}"
VENV_DIR="${DD_PREP_VENV:?}"

module purge
module load StdEnv/2023
module load python/3.11
module load gcc rdkit

source "$VENV_DIR/bin/activate"

echo "========================================"
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Time:   $(date)"
echo "========================================"

dd-prep --config "$CONFIG" --step organize
dd-prep --config "$CONFIG" --step fingerprint

echo "Job 4 complete at $(date)"