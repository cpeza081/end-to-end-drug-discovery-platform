#!/bin/bash
#SBATCH --job-name=dd_flipper
#SBATCH --time=03:00:00          # Adjust: ~2h per 10M molecules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1        # flipper is single-threaded
#SBATCH --mem=8G
#SBATCH --array=0-999%20         # Up to 1000 chunks; max 20 running at once
                                 # The %20 throttle protects the scheduler.
                                 # Tasks with index >= actual chunk count exit cleanly.
#SBATCH --output=logs/02_flipper_%A_%a.log   # %A = job ID, %a = array index
#SBATCH --account=rrg-checco89    # ← CHANGE THIS

# =============================================================================
# Job 2: Stereoisomer enumeration with OpenEye FLIPPER (array job).
#
# Each array task processes exactly one chunk file.  The --chunk-index flag
# maps SLURM_ARRAY_TASK_ID (0-based) to the sorted list of chunk files.
# Tasks whose index exceeds the chunk count exit cleanly — this is why the
# array size (999) can safely exceed the actual number of chunks.
#
# Input:  work_dir/smiles/smiles_all_NNN.smi
# Output: work_dir/smiles/smiles_all_NNN_isom.smi
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?}"
VENV_DIR="${DD_PREP_VENV:?}"

module purge
module load StdEnv/2023
module load python/3.11
module load gcc rdkit

source "$VENV_DIR/bin/activate"

# OpenEye licence — must be visible on all compute nodes.
export OE_LICENSE="${OE_LICENSE:-$HOME/oe_license.txt}"

echo "========================================"
echo "Job:         $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Time:        $(date)"
echo "========================================"

dd-prep --config "$CONFIG" \
        --step flipper \
        --chunk-index "$SLURM_ARRAY_TASK_ID"

echo "Task $SLURM_ARRAY_TASK_ID complete at $(date)"