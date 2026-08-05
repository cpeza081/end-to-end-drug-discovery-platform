#!/bin/bash
#SBATCH --job-name=dd_filter_split
#SBATCH --time=1:00:00          # Filter is the throughput bottleneck. RDKit
                                 # parse + descriptors runs at a few thousand
                                 # mol/s PER CORE, so even parallelised across
                                 # 32 cores a ~1B molecule library is many hours.
                                 # Scale this with (library size / cores);
                                 # 4h is far too short and the job will time out.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32       # Filter now fans RDKit work across processes.
                                 # Set filter.n_workers in the config to match.
#SBATCH --mem=32G
#SBATCH --output=logs/01_filter_split_%j.log
#SBATCH --account=def-YOURPI

# =============================================================================
# Job 1: Filter and split the raw SMILES library.
# Account and other settings are updated automatically by setup_cluster.sh
#
# Note: the filter step is now parallelised within the
# step (filter.n_workers), but it is still CPU-bound — size --time and
# --cpus-per-task to your library. For very large libraries consider raising
# both, or pre-splitting the raw file and running filter as an array job.
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?Set DD_PREP_CONFIG or run setup_cluster.sh}"
VENV_DIR="${DD_PREP_VENV:?Set DD_PREP_VENV or run setup_cluster.sh}"

module purge
module load StdEnv/2023
module load python/3.11
# scipy-stack supplies pandas/numpy from CVMFS. On Alliance pip is
# pointed at a local wheelhouse, and pandas is expected to
# come from this module.
module load scipy-stack
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