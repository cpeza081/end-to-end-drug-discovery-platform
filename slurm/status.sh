#!/bin/bash
# =============================================================================
# status.sh — Show the current state of the dd_prep pipeline
#
# Usage:
#   bash slurm/status.sh [--config /path/to/config.yaml]
#
# Shows:
#   • Which pipeline stages have completed
#   • Molecule counts at each stage
#   • Currently running/pending SLURM jobs
#   • Any errors in recent logs
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; GREY='\033[0;37m'; BOLD='\033[1m'; RESET='\033[0m'

done_()    { echo -e "  ${GREEN}✓${RESET}  $*"; }
pending_() { echo -e "  ${YELLOW}○${RESET}  $*"; }
failed_()  { echo -e "  ${RED}✗${RESET}  $*"; }
skip_()    { echo -e "  ${GREY}—${RESET}  $*"; }

# ── Load config ───────────────────────────────────────────────────────────────
ENV_FILE="$PROJECT_DIR/.dd_prep_env"
CONFIG=""

# Parse --config argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

[ -f "$ENV_FILE" ] && source "$ENV_FILE"
CONFIG="${CONFIG:-${DD_PREP_CONFIG:-}}"

if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    # Try to find a config file
    CONFIG=$(find "$PROJECT_DIR" -maxdepth 1 -name "*.yaml" ! -name "config_example.yaml" | head -1)
fi

# Read work_dir from config
WORK_DIR=""
if [ -n "$CONFIG" ] && [ -f "$CONFIG" ]; then
    WORK_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('work_dir', ''))
" 2>/dev/null || true)
fi

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   Deep Docking Prep — Pipeline Status${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Config  : ${CONFIG:-not found}"
echo "  Outputs : ${WORK_DIR:-unknown}"
echo ""

# ── Check output files ────────────────────────────────────────────────────────
echo -e "${BOLD}Pipeline stages:${RESET}"
echo ""

if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then

    # Filter
    FILTERED="$WORK_DIR/filtered/library_filtered.smi"
    if [ -f "$FILTERED" ]; then
        N=$(( $(wc -l < "$FILTERED") - 1 ))
        done_ "Filter          $N molecules"
    else
        pending_ "Filter          not yet run"
    fi

    # Split
    SMILES_DIR="$WORK_DIR/smiles"
    if [ -d "$SMILES_DIR" ]; then
        N_CHUNKS=$(ls "$SMILES_DIR"/smiles_all_*.smi 2>/dev/null | grep -v "_isom\|_states" | wc -l)
        if [ "$N_CHUNKS" -gt 0 ]; then
            done_ "Split           $N_CHUNKS chunk(s)"
        else
            pending_ "Split           not yet run"
        fi
    else
        pending_ "Split           not yet run"
    fi

    # Flipper
    N_ISOM=$(ls "$SMILES_DIR"/smiles_all_*_isom.smi 2>/dev/null | wc -l)
    if [ "$N_ISOM" -gt 0 ]; then
        done_ "Flipper         $N_ISOM chunk(s) processed"
    else
        pending_ "Flipper         not yet run"
    fi

    # Tautomers
    N_STATES=$(ls "$SMILES_DIR"/smiles_all_*_states.smi 2>/dev/null | wc -l)
    if [ "$N_STATES" -gt 0 ]; then
        done_ "Tautomers       $N_STATES chunk(s) processed"
    else
        pending_ "Tautomers       not yet run"
    fi

    # Organize
    LIB_DIR="$WORK_DIR/library_prepared"
    if [ -d "$LIB_DIR" ]; then
        N_PREP=$(ls "$LIB_DIR"/*.txt 2>/dev/null | wc -l)
        if [ "$N_PREP" -gt 0 ]; then
            TOTAL_MOLS=$(cat "$LIB_DIR"/*.txt 2>/dev/null | wc -l)
            done_ "Organize        $N_PREP file(s), $TOTAL_MOLS molecules total"
        else
            pending_ "Organize        not yet run"
        fi
    else
        pending_ "Organize        not yet run"
    fi

    # Fingerprints
    FP_DIR="$WORK_DIR/library_prepared_fp"
    if [ -d "$FP_DIR" ]; then
        N_FP=$(ls "$FP_DIR"/*.txt 2>/dev/null | wc -l)
        if [ "$N_FP" -gt 0 ]; then
            done_ "Fingerprints    $N_FP file(s) — ready for Deep Docking"
        else
            pending_ "Fingerprints    not yet run"
        fi
    else
        pending_ "Fingerprints    not yet run"
    fi

else
    echo "  Output directory not found. Has the pipeline been submitted?"
fi

# ── SLURM jobs ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Active SLURM jobs:${RESET}"
echo ""

JOBS=$(squeue -u $USER --format="  %-18i %-16j %-4t %-12M %R" --noheader 2>/dev/null \
       | grep -i "dd_\|coordinator" || true)

if [ -n "$JOBS" ]; then
    echo "$JOBS"
else
    echo "  No active jobs."
fi

# ── Recent errors ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Recent errors (last 24h):${RESET}"
echo ""

LOG_DIR="$PROJECT_DIR/logs"
if [ -d "$LOG_DIR" ]; then
    ERRORS=$(find "$LOG_DIR" -name "*.log" -newer "$LOG_DIR" -mmin -1440 \
             -exec grep -l "ERROR\|Error\|FAILED\|ModuleNotFoundError" {} \; 2>/dev/null \
             | head -5 || true)
    if [ -n "$ERRORS" ]; then
        while IFS= read -r f; do
            failed_ "$f"
            grep -m 2 "ERROR\|Error\|FAILED" "$f" | sed 's/^/       /'
        done <<< "$ERRORS"
    else
        echo "  No errors found in recent logs."
    fi
else
    echo "  No logs directory found."
fi

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo ""