#!/bin/bash
#SBATCH --job-name=dd_tautomers
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-999%20
#SBATCH --output=logs/03_tautomers_%A_%a.log
#SBATCH --account=rrg-checco89    # ← CHANGE THIS

# =============================================================================
# Job 3: Protonation state assignment with OpenEye TAUTOMERS (array job).
#
# Same array structure as Job 2.  Each task takes the _isom.smi file
# produced by flipper for the same chunk index and applies TAUTOMERS.
#
# Input:  work_dir/smiles/smiles_all_NNN_isom.smi
# Output: work_dir/smiles/smiles_all_NNN_states.smi
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?}"
VENV_DIR="${DD_PREP_VENV:?}"

module purge
module load StdEnv/2023
module load python/3.11

source "$VENV_DIR/bin/activate"
export OE_LICENSE="${OE_LICENSE:-$HOME/oe_license.txt}"

echo "========================================"
echo "Job:         $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Time:        $(date)"
echo "========================================"

dd-prep --config "$CONFIG" \
        --step tautomer \
        --chunk-index "$SLURM_ARRAY_TASK_ID"

echo "Task $SLURM_ARRAY_TASK_ID complete at $(date)"