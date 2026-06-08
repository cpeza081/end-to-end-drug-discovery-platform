#!/bin/bash
#SBATCH --job-name=dd_omega
#SBATCH --time=08:00:00          # 3D generation is slow — adjust generously
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Passed to oeomega via -mpi_np in config
#SBATCH --mem=16G
#SBATCH --array=0-999%10         # Throttle to 10 simultaneous — omega is memory-hungry
#SBATCH --output=logs/05_omega_%A_%a.log
#SBATCH --account=rrg-checco89    # ← CHANGE THIS

# =============================================================================
# Job 5 (optional): 3-D conformer generation with OpenEye OMEGA (array job).
#
# Only needed if you want the full library in 3-D.  For a standard DD run,
# OMEGA is applied per-iteration on the sampled subsets, not on the full
# library.  This job is NOT submitted by submit_pipeline.sh by default.
#
# Input:  work_dir/library_prepared/smiles_all_NNN.txt
# Output: work_dir/sdf/smiles_all_NNN.sdf.gz
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
        --step omega \
        --chunk-index "$SLURM_ARRAY_TASK_ID"

echo "Task $SLURM_ARRAY_TASK_ID complete at $(date)"