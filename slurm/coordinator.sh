#!/bin/bash
#SBATCH --job-name=dd_coordinator
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/coordinator_%j.log
#SBATCH --account=def-YOURPI

# =============================================================================
# coordinator.sh — Counts chunks after split and submits right-sized array jobs
#
# This job runs after filter+split completes. It reads the actual number of
# chunk files on disk and submits the flipper/tautomers arrays with exactly
# the right size — no more waiting for 999 no-op tasks to drain.
# =============================================================================

set -euo pipefail

CONFIG="${DD_PREP_CONFIG:?}"
VENV_DIR="${DD_PREP_VENV:?}"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Read work_dir from config
WORK_DIR=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('work_dir', ''))
" 2>/dev/null)

if [ -z "$WORK_DIR" ]; then
    echo "ERROR: Could not read work_dir from config."
    exit 1
fi

echo "================================================"
echo "Coordinator job: $SLURM_JOB_ID"
echo "Work directory:  $WORK_DIR"
echo "Time:            $(date)"
echo "================================================"

# Count actual chunk files
SMILES_DIR="$WORK_DIR/smiles"
N_CHUNKS=$(ls "$SMILES_DIR"/smiles_all_*.smi 2>/dev/null | wc -l)

if [ "$N_CHUNKS" -eq 0 ]; then
    echo "ERROR: No chunk files found in $SMILES_DIR"
    echo "The split step may have failed. Check logs/01_filter_split_*.log"
    exit 1
fi

ARRAY_MAX=$((N_CHUNKS - 1))
echo "Found $N_CHUNKS chunk(s) — submitting arrays 0-$ARRAY_MAX"

EXPORTS="DD_PREP_CONFIG=$CONFIG,DD_PREP_VENV=$VENV_DIR"

# Submit flipper array with exact size
JOB2=$(sbatch \
    --export="$EXPORTS" \
    --array="0-$ARRAY_MAX" \
    --parsable \
    "$SCRIPT_DIR/02_flipper.sh")
echo "Submitted flipper array (0-$ARRAY_MAX): $JOB2"

# Submit tautomers array, depends on all flipper tasks
JOB3=$(sbatch \
    --export="$EXPORTS" \
    --array="0-$ARRAY_MAX" \
    --dependency="aftercorr:$JOB2" \
    --parsable \
    "$SCRIPT_DIR/03_tautomers.sh")
echo "Submitted tautomers array (0-$ARRAY_MAX): $JOB3  [after $JOB2]"

# Submit organize + fingerprints
JOB4=$(sbatch \
    --export="$EXPORTS" \
    --dependency="aftercorr:$JOB3" \
    --parsable \
    "$SCRIPT_DIR/04_organize_fp.sh")
echo "Submitted organize+fp: $JOB4  [after all of $JOB3]"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  bash slurm/status.sh"