#!/bin/bash
# =============================================================================
# submit_pipeline.sh — Submit the full DD prep pipeline to SLURM.
#
# Usage:
#   bash slurm/submit_pipeline.sh --config /path/to/config.yaml
#
# This script submits jobs 1–4 with proper SLURM dependencies so they run
# in the correct order automatically:
#
#   Job 1 (filter + split)    ──┐
#   Job 2 (flipper array)     ──┤  afterok:1
#   Job 3 (tautomers array)   ──┤  afterok:2
#   Job 4 (organize + FP)     ──┘  afterok:3
#
# Job 5 (omega) is submitted separately when needed — see --omega flag.
#
# Requirements:
#   • Run 00_setup_env.sh once before this script.
#   • Set DD_PREP_VENV in your environment or edit VENV_DIR below.
#   • Edit the --account line in each .sh file to match your PI's account.
#
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
VENV_DIR="${DD_PREP_VENV:-$SCRATCH/dd_prep_venv}"
SUBMIT_OMEGA=false
CONFIG=""
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG="$2";      shift 2 ;;
        --venv)     VENV_DIR="$2";    shift 2 ;;
        --omega)    SUBMIT_OMEGA=true; shift ;;
        --help|-h)
            echo "Usage: $0 --config /path/to/config.yaml [--venv /path/to/venv] [--omega]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required."
    echo "Usage: $0 --config /path/to/config.yaml"
    exit 1
fi

CONFIG="$(realpath "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    echo "Run slurm/00_setup_env.sh first."
    exit 1
fi

# ── Create logs directory ─────────────────────────────────────────────────────
mkdir -p logs

# ── Export variables for all jobs ─────────────────────────────────────────────
# sbatch --export passes these into each job's environment.
EXPORTS="DD_PREP_CONFIG=$CONFIG,DD_PREP_VENV=$VENV_DIR"

echo "=============================================="
echo "Submitting DD prep pipeline"
echo "  Config : $CONFIG"
echo "  Venv   : $VENV_DIR"
echo "  Omega  : $SUBMIT_OMEGA"
echo "=============================================="

# ── Job 1: Filter + Split ─────────────────────────────────────────────────────
JOB1=$(sbatch \
    --export="$EXPORTS" \
    --parsable \
    "$SLURM_DIR/01_filter_split.sh")
echo "Submitted Job 1 (filter+split):   $JOB1"

# ── Job 2: Flipper array ──────────────────────────────────────────────────────
JOB2=$(sbatch \
    --export="$EXPORTS" \
    --dependency="afterok:$JOB1" \
    --parsable \
    "$SLURM_DIR/02_flipper.sh")
echo "Submitted Job 2 (flipper array):   $JOB2  [after $JOB1]"

# ── Job 3: Tautomers array ────────────────────────────────────────────────────
JOB3=$(sbatch \
    --export="$EXPORTS" \
    --dependency="afterok:$JOB2" \
    --parsable \
    "$SLURM_DIR/03_tautomers.sh")
echo "Submitted Job 3 (tautomers array): $JOB3  [after $JOB2]"

# ── Job 4: Organize + Fingerprints ───────────────────────────────────────────
# aftercorr:JOB3 means "after all array tasks of JOB3 have completed".
JOB4=$(sbatch \
    --export="$EXPORTS" \
    --dependency="aftercorr:$JOB3" \
    --parsable \
    "$SLURM_DIR/04_organize_fp.sh")
echo "Submitted Job 4 (organize+fp):     $JOB4  [after all of $JOB3]"

# ── Job 5: Omega (optional) ───────────────────────────────────────────────────
if $SUBMIT_OMEGA; then
    JOB5=$(sbatch \
        --export="$EXPORTS" \
        --dependency="afterok:$JOB4" \
        --parsable \
        "$SLURM_DIR/05_omega.sh")
    echo "Submitted Job 5 (omega array):     $JOB5  [after $JOB4]"
fi

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/01_filter_split_${JOB1}.log"
echo ""
echo "To cancel everything:"
echo "  scancel $JOB1 $JOB2 $JOB3 $JOB4"