#!/bin/bash
# =============================================================================
# submit_pipeline.sh — Submit the full DD prep pipeline to SLURM
#
# Usage:
#   bash slurm/submit_pipeline.sh [--config /path/to/config.yaml]
#
# If no --config is given, uses the config path saved by setup_cluster.sh.
#
# Job flow:
#   Job 1 (filter + split)   runs first
#   Job 2 (coordinator)      waits for job 1, counts chunks, submits:
#     Job 3 (flipper array)     — one task per chunk
#     Job 4 (tautomers array)   — waits for flipper
#     Job 5 (organize + fp)     — waits for tautomers
#
# This means array sizes are always exactly right — no wasted tasks.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load saved environment from setup_cluster.sh ─────────────────────────────
ENV_FILE="$PROJECT_DIR/.dd_prep_env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "ERROR: .dd_prep_env not found."
    echo "Please run setup first:  bash slurm/setup_cluster.sh"
    exit 1
fi

# ── Parse arguments ───────────────────────────────────────────────────────────
CONFIG="${DD_PREP_CONFIG:-}"
SUBMIT_OMEGA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG="$2";      shift 2 ;;
        --venv)      DD_PREP_VENV="$2"; shift 2 ;;
        --omega)     SUBMIT_OMEGA=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--config /path/to/config.yaml] [--omega]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [ -z "$CONFIG" ]; then
    echo "ERROR: No config file specified."
    echo "Run setup_cluster.sh first, or pass --config /path/to/config.yaml"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

if [ -z "${DD_PREP_VENV:-}" ]; then
    echo "ERROR: DD_PREP_VENV not set. Run setup_cluster.sh first."
    exit 1
fi

if [ ! -d "$DD_PREP_VENV" ]; then
    echo "ERROR: Virtual environment not found: $DD_PREP_VENV"
    echo "Run setup_cluster.sh to recreate it."
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs"

EXPORTS="DD_PREP_CONFIG=$CONFIG,DD_PREP_VENV=$DD_PREP_VENV,DD_PREP_PROJECT=$DD_PREP_PROJECT"

echo "=============================================="
echo "  Deep Docking preparation pipeline"
echo "  Config : $CONFIG"
echo "  Venv   : $DD_PREP_VENV"
echo "=============================================="

# ── Job 1: Filter + Split ─────────────────────────────────────────────────────
JOB1=$(sbatch \
    --export="$EXPORTS" \
    --parsable \
    "$SCRIPT_DIR/01_filter_split.sh")
echo "Submitted Job 1 (filter+split):    $JOB1"

# ── Job 2: Coordinator — waits for split, then submits right-sized arrays ─────
JOB2=$(sbatch \
    --export="$EXPORTS" \
    --dependency="afterok:$JOB1" \
    --parsable \
    "$SCRIPT_DIR/coordinator.sh")
echo "Submitted Job 2 (coordinator):     $JOB2  [after $JOB1]"

echo ""
echo "The coordinator will automatically submit flipper, tautomers,"
echo "and fingerprint jobs with the correct array sizes once the"
echo "split step finishes."
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  bash slurm/status.sh"
echo ""
echo "To cancel everything:"
echo "  scancel -u \$USER"