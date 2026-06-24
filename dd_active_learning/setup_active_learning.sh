#!/bin/bash
# =============================================================================
# setup_active_learning.sh
# 
# Interactive one-time setup wizard for dd_active_learning
#
# Usage:
#   bash dd_active_learning/setup_active_learning.sh
#
# Mirrors the UX of slurm/setup_cluster.sh (dd_prep's wizard) so both halves
# of the platform feel like one product: same colours, same spinner, same
# auto-detect-then-ask-manually fallback pattern.
#
# Press Ctrl+C at any time to cancel cleanly.
# =============================================================================

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
ask()     { echo -e "\n${BOLD}$*${RESET}"; }

# ── Ctrl+C / cancellation ─────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo ""
    warn "Setup cancelled by user."
    kill $(jobs -p) 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── Spinner ───────────────────────────────────────────────────────────────────
# Same animation as slurm/setup_cluster.sh.
spinner() {
    local pid=$1
    local label="$2"
    local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${BLUE}%s${RESET}  %s" "${frames[$i]}" "$label"
        i=$(( (i + 1) % ${#frames[@]} ))
        sleep 0.1
    done
    printf "\r%-60s\r" " "
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   Deep Docking Active Learning Setup Wizard${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Press Ctrl+C at any time to cancel."
echo "  Press Enter to accept defaults shown in [brackets]."
echo ""

# ── Step 1: SLURM account ─────────────────────────────────────────────────────
info "Detecting your SLURM accounts..."

ACCOUNTS=""

if command -v sacctmgr &>/dev/null; then # sacctmgr is the canonical way to get account info, but it may not be on PATH on a compute node, so check first before trying to run it
    sacctmgr show associations user=$USER format=account%30 --noheader > /tmp/dd_al_accounts.txt 2>/tmp/dd_al_accounts.err &
    spinner $! "Querying SLURM..."
    wait $!

    if [ -s /tmp/dd_al_accounts.err ]; then
        warn "sacctmgr reported an error:"
        cat /tmp/dd_al_accounts.err | sed 's/^/         /'
    else
        ACCOUNTS=$(cat /tmp/dd_al_accounts.txt | tr -d ' ' | grep -v "^$" | sort -u)
    fi
else
    warn "sacctmgr not found on PATH. This doesn't look like a SLURM login node."
    warn "You can still write a config now and fill in the account later."
fi

if [ -z "$ACCOUNTS" ]; then
    warn "No SLURM accounts detected automatically."
    ask "Enter your billing account name manually (or press Enter to fill in later):"
    read -r SLURM_ACCOUNT
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-CHANGE_ME}"
    [ "$SLURM_ACCOUNT" = "CHANGE_ME" ] && warn "Remember to set 'account:' in campaign.yaml before launching."
else
    echo ""
    echo "  Available accounts:"
    i=1
    declare -a ACCOUNT_LIST
    while IFS= read -r acc; do
        echo "    $i) $acc"
        ACCOUNT_LIST[$i]="$acc"
        ((i++))
    done <<< "$ACCOUNTS"

    ask "Which account should jobs be billed to? Enter number [1]:"
    read -r ACCOUNT_CHOICE
    ACCOUNT_CHOICE="${ACCOUNT_CHOICE:-1}"
    SLURM_ACCOUNT="${ACCOUNT_LIST[$ACCOUNT_CHOICE]}"

    if [ -z "$SLURM_ACCOUNT" ]; then
        error "Invalid choice."
        exit 1
    fi
fi
success "Using account: $SLURM_ACCOUNT"

# ── Step 2: Library source ────────────────────────────────────────────────────
# This is the dd_prep to dd_active_learning bridge point. A campaign needs
# library.smiles_dir and library.fingerprint_dir, and the most common way to get
# those is from a finished dd_prep run, so we offer that path directly here.
ask "How will the prepared library be provided?"
echo "    1) Link an existing finished dd_prep run (recommended)"
echo "    2) I'll set library.smiles_dir / library.fingerprint_dir manually later"
read -r LIBRARY_CHOICE
LIBRARY_CHOICE="${LIBRARY_CHOICE:-1}"

SMILES_DIR=""
FP_DIR=""

if [ "$LIBRARY_CHOICE" = "1" ]; then
    ask "Path to the dd_prep work_dir (or its config.yaml):"
    read -r PREP_SOURCE

    if [ -f "$PREP_SOURCE" ]; then
        # Looks like a config file, so read work_dir out of it
        PREP_WORK_DIR=$(python3 -c "
import yaml
with open('$PREP_SOURCE') as f:
    print((yaml.safe_load(f) or {}).get('work_dir', ''))
" 2>/dev/null)
    else
        PREP_WORK_DIR="$PREP_SOURCE"
    fi

    if [ -d "$PREP_WORK_DIR/library_prepared" ] && [ -d "$PREP_WORK_DIR/library_prepared_fp" ]; then
        SMILES_DIR="$(cd "$PREP_WORK_DIR/library_prepared" && pwd)"
        FP_DIR="$(cd "$PREP_WORK_DIR/library_prepared_fp" && pwd)"
        success "Found prepared library at $PREP_WORK_DIR"
    else
        warn "Could not find library_prepared/ and library_prepared_fp/ under $PREP_WORK_DIR"
        warn "You can link this later with: python dd_link.py --prep-work-dir $PREP_WORK_DIR --campaign <config>"
        SMILES_DIR="\$SCRATCH/library_prepared"
        FP_DIR="\$SCRATCH/library_prepared_fp"
    fi
else
    SMILES_DIR="\$SCRATCH/library_prepared"
    FP_DIR="\$SCRATCH/library_prepared_fp"
    info "Placeholder paths written. Update library.smiles_dir / library.fingerprint_dir, or run dd_link.py once dd_prep finishes."
fi

# ── Step 3: Campaign identity ─────────────────────────────────────────────────
ask "Campaign name (used for directories and job names):"
read -r CAMPAIGN_NAME
CAMPAIGN_NAME="${CAMPAIGN_NAME:-my_target_dd}"

DEFAULT_CAMPAIGN_DIR="\$SCRATCH/dd_campaigns/$CAMPAIGN_NAME"
ask "Where should campaign outputs go? [$DEFAULT_CAMPAIGN_DIR]:"
read -r CAMPAIGN_DIR_INPUT
CAMPAIGN_DIR_RAW="${CAMPAIGN_DIR_INPUT:-$DEFAULT_CAMPAIGN_DIR}"
# CAMPAIGN_DIR_RAW may contain an unexpanded $SCRATCH (e.g. from the default
# above). We expand it now so every later mkdir/file path is a real
# path, while CAMPAIGN_DIR_RAW (with $SCRATCH literal) is what gets written
# into campaign.yaml.
CAMPAIGN_DIR_EXPANDED=$(eval echo "$CAMPAIGN_DIR_RAW")
success "Campaign output directory: $CAMPAIGN_DIR_RAW"

# ── Step 4: DD protocol (auto-find or clone) ──────────────────────────────────
# This step reuses the same find+spinner pattern setup_cluster.sh
# uses for locating OpenEye.
info "Looking for an existing DD_protocol checkout..."

find "$HOME" "$SCRATCH" /project /scratch 2>/dev/null \
    -maxdepth 6 -type d -name "DD_protocol" \
    > /tmp/dd_al_protocol_candidates.txt 2>/dev/null &
spinner $! "Searching filesystem for DD_protocol..."
wait $! || true

DD_PROTOCOL_DIR=""
REQUIRED_SCRIPTS=(
    "scripts_1/molecular_file_count_updated.py"
    "scripts_1/sampling.py"
    "scripts_1/sanity_check.py"
    "scripts_1/extracting_morgan.py"
    "scripts_1/extracting_smiles.py"
    "scripts_2/extract_labels.py"
    "scripts_2/simple_job_models_manual.py"
    "scripts_2/hyperparameter_result_evaluation.py"
    "scripts_2/simple_job_predictions_manual.py"
    "utilities/final_extraction.py"
)

# A directory named DD_protocol isn't proof it's a valid one,
# so check that every script dd_orchestrator.py actually calls is present. 
# Same script list dd_validate.py checks.
validate_dd_protocol_dir() {
    local candidate="$1"
    for script in "${REQUIRED_SCRIPTS[@]}"; do
        [ -f "$candidate/$script" ] || return 1
    done
    return 0
}

while IFS= read -r candidate; do
    [ -z "$candidate" ] && continue
    if validate_dd_protocol_dir "$candidate"; then
        DD_PROTOCOL_DIR="$candidate"
        break
    fi
done < /tmp/dd_al_protocol_candidates.txt

if [ -n "$DD_PROTOCOL_DIR" ]; then
    success "Found a valid DD_protocol checkout: $DD_PROTOCOL_DIR"
else
    warn "No valid DD_protocol checkout found automatically."
    ask "Clone it now from https://github.com/jamesgleave/DD_protocol? (Y/n):"
    read -r DO_CLONE

    if [[ ! "$DO_CLONE" =~ ^[Nn]$ ]]; then
        DEFAULT_CLONE_DIR="\$SCRATCH/DD_protocol"
        ask "Clone destination [$DEFAULT_CLONE_DIR]:"
        read -r CLONE_DIR_INPUT
        CLONE_DIR="${CLONE_DIR_INPUT:-$DEFAULT_CLONE_DIR}"
        CLONE_DIR_EXPANDED=$(eval echo "$CLONE_DIR")

        if [ -d "$CLONE_DIR_EXPANDED" ]; then
            warn "Directory already exists: $CLONE_DIR_EXPANDED"
            if validate_dd_protocol_dir "$CLONE_DIR_EXPANDED"; then
                success "Existing directory is a valid DD_protocol checkout, so it will be used."
                DD_PROTOCOL_DIR="$CLONE_DIR_EXPANDED"
            else
                error "Existing directory is not empty and is not a valid DD_protocol checkout."
                error "Remove it or choose a different destination, then re-run this wizard."
            fi
        else
            git clone https://github.com/jamesgleave/DD_protocol "$CLONE_DIR_EXPANDED" &
            spinner $! "Cloning DD_protocol..."
            if wait $!; then
                if validate_dd_protocol_dir "$CLONE_DIR_EXPANDED"; then
                    success "Cloned and verified: $CLONE_DIR_EXPANDED"
                    DD_PROTOCOL_DIR="$CLONE_DIR_EXPANDED"
                else
                    error "Clone succeeded but expected scripts are missing. "
                    error "The upstream repository layout may have changed, so check $CLONE_DIR_EXPANDED manually."
                fi
            else
                error "git clone failed. Check network access and the URL above."
            fi
        fi
    fi

    if [ -z "$DD_PROTOCOL_DIR" ]; then
        warn "Continuing without a verified DD_protocol checkout."
        ask "Enter the path manually (or press Enter to fill in later):"
        read -r DD_PROTOCOL_DIR_INPUT
        DD_PROTOCOL_DIR="${DD_PROTOCOL_DIR_INPUT:-\$SCRATCH/DD_protocol}"
    fi
fi

# ── Step 5: OpenEye ───────────────────────────────────────────────────────────
# Same auto-detect-then-manual-fallback pattern as setup_cluster.sh, kept separate per wizard.
echo ""
info "OpenEye configuration"
echo "  OpenEye is required for OMEGA (ligand prep) and FRED (docking)."
echo ""

ask "Auto-detect OpenEye installation? This searches the filesystem and may take 1-2 minutes. (Y/n):"
read -r OE_AUTO
OE_BIN=""
OE_LIC=""

if [[ ! "$OE_AUTO" =~ ^[Nn]$ ]]; then
    info "Searching for OpenEye binaries..."
    find /project /opt /software 2>/dev/null \
        -name "oeomega" -not -path "*/arch/*" \
        > /tmp/dd_al_oe_bin.txt 2>/dev/null &
    spinner $! "Searching filesystem for OpenEye (Ctrl+C to cancel and enter manually)..."
    wait $! || true
    OE_BIN=$(head -1 /tmp/dd_al_oe_bin.txt | xargs -I{} dirname {} 2>/dev/null || true)

    if [ -n "$OE_BIN" ]; then
        success "Found OpenEye binaries: $OE_BIN"
    else
        warn "Could not find OpenEye binaries automatically."
        ask "Enter path to OpenEye bin directory manually (or press Enter to skip):"
        read -r OE_BIN_INPUT
        OE_BIN="${OE_BIN_INPUT:-}"
    fi

    info "Searching for OpenEye licence..."
    find /project /opt /home/$USER 2>/dev/null \
        -name "oe_license.txt" -not -path "*/arch/*" -not -name "*.bak*" \
        > /tmp/dd_al_oe_lic.txt 2>/dev/null &
    spinner $! "Searching for licence file (Ctrl+C to cancel and enter manually)..."
    wait $! || true
    OE_LIC=$(head -1 /tmp/dd_al_oe_lic.txt || true)

    if [ -n "$OE_LIC" ]; then
        success "Found OpenEye licence: $OE_LIC"
    else
        warn "Could not find licence file automatically."
        ask "Enter path to oe_license.txt manually (or press Enter to skip):"
        read -r OE_LIC_INPUT
        OE_LIC="${OE_LIC_INPUT:-}"
    fi
else
    ask "Path to OpenEye bin directory (contains oeomega, fred):"
    read -r OE_BIN_INPUT
    OE_BIN="${OE_BIN_INPUT:-}"

    ask "Path to oe_license.txt:"
    read -r OE_LIC_INPUT
    OE_LIC="${OE_LIC_INPUT:-}"
fi

# ── Step 6: Conda environment ─────────────────────────────────────────────────
ask "Name of the conda environment with rdkit / tensorflow / DD dependencies [dd-env]:"
read -r CONDA_ENV_INPUT
CONDA_ENV="${CONDA_ENV_INPUT:-dd-env}"

if command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "^${CONDA_ENV} \|^${CONDA_ENV}\$"; then
    success "Conda environment found: $CONDA_ENV"
else
    warn "Conda environment '$CONDA_ENV' not found (or conda not available right now)."
    warn "Make sure it exists before launching a campaign. See DD_protocol's environment.yml."
fi

# ── Step 7: Docking program ───────────────────────────────────────────────────
ask "Which docking program will this campaign use? (1=FRED, 2=GLIDE) [1]:"
read -r DOCK_CHOICE
DOCK_CHOICE="${DOCK_CHOICE:-1}"
if [ "$DOCK_CHOICE" = "2" ]; then
    DOCK_PROGRAM="GLIDE"
    DOCK_SCORE_KEYWORD="r_i_docking_score"
else
    DOCK_PROGRAM="FRED"
    DOCK_SCORE_KEYWORD="FRED Chemgauss4 score"
fi

ask "Path to the docking grid file:"
read -r GRID_FILE_INPUT
GRID_FILE="${GRID_FILE_INPUT:-\$SCRATCH/receptor/${DOCK_PROGRAM,,}_grid.oeb}"

# ── Step 8: Write campaign.yaml ───────────────────────────────────────────────
mkdir -p "$CAMPAIGN_DIR_EXPANDED"
CONFIG_FILE="$CAMPAIGN_DIR_EXPANDED/campaign.yaml"

if [ -f "$CONFIG_FILE" ]; then
    ask "campaign.yaml already exists. Overwrite? (y/N):"
    read -r OW
    [[ "$OW" =~ ^[Yy]$ ]] || CONFIG_FILE="$CAMPAIGN_DIR_EXPANDED/campaign_$(date +%Y%m%d_%H%M%S).yaml"
fi

if ! cat > "$CONFIG_FILE" << YAML
# =============================================================================
# Deep Docking Active Learning Campaign Configuration
# Generated by setup_active_learning.sh on $(date)
# =============================================================================

campaign_name: "$CAMPAIGN_NAME"
project_dir: "$CAMPAIGN_DIR_RAW"

library:
  smiles_dir: "$SMILES_DIR"
  fingerprint_dir: "$FP_DIR"

docking:
  program: "$DOCK_PROGRAM"
  grid_file: "$GRID_FILE"
  score_keyword: "$DOCK_SCORE_KEYWORD"
  glide_template: ""

dd:
  total_iterations: 11
  train_size: 1000000
  val_size: 1000000
  percent_first: 1.0
  percent_last: 0.01
  recall: 0.90
  num_models: 24
  num_cpus_sampling: 60

scheduler:
  type: "SLURM"
  account: "$SLURM_ACCOUNT"
  cpu_partition: "cpu"
  gpu_partition: "gpu"

  walltime:
    phase1_sampling: "00:30:00"
    phase2_ligand_prep: "08:00:00"
    phase3_docking: "24:00:00"
    phase4_training: "20:00:00"
    phase5_inference: "06:00:00"
    final_extraction: "02:00:00"

  resources:
    phase1_sampling:  {nodes: 1, cpus: 60, mem: "32G", gpus: 0}
    phase2_ligand_prep: {nodes: 3, cpus: 60, mem: "32G", gpus: 0}
    phase3_docking:   {nodes: 1, cpus: 60, mem: "64G", gpus: 0}
    phase4_training:  {nodes: 1, cpus: 8,  mem: "32G", gpus: 1}
    phase5_inference: {nodes: 1, cpus: 8,  mem: "32G", gpus: 1}
    final_extraction: {nodes: 1, cpus: 60, mem: "32G", gpus: 0}

env:
  conda_env: "$CONDA_ENV"
  openeye_dir: "$OE_BIN"
  dd_protocol_dir: "$DD_PROTOCOL_DIR"
YAML
then
    error "Failed to write $CONFIG_FILE. Check that $CAMPAIGN_DIR_EXPANDED is writable."
    exit 1
fi

if [ -n "$OE_LIC" ]; then
    echo "  # OE_LICENSE auto-detected during setup:" >> "$CONFIG_FILE"
    echo "  # $OE_LIC" >> "$CONFIG_FILE"
fi

success "Config written to: $CONFIG_FILE"

# ── Step 9: Validate ──────────────────────────────────────────────────────────
echo ""
info "Running validation..."
python3 "$SCRIPT_DIR/dd_validate.py" --config "$CONFIG_FILE" 2>&1 || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}   All done!${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Account       : $SLURM_ACCOUNT"
echo "  Config        : $CONFIG_FILE"
echo "  DD protocol   : ${DD_PROTOCOL_DIR:-not set}"
[ -n "$OE_LIC" ] && echo "  OE Lic        : $OE_LIC"
echo ""
if [ "$LIBRARY_CHOICE" != "1" ] || [ "$SMILES_DIR" = "\$SCRATCH/library_prepared" ]; then
    echo "  Before launching, link a prepared library:"
    echo -e "    ${BOLD}python dd_active_learning/dd_link.py --prep-work-dir <dd_prep work_dir> --campaign $CONFIG_FILE${RESET}"
    echo ""
fi
echo "  Review the config, then launch:"
echo -e "    ${BOLD}python dd_active_learning/dd_orchestrator.py --config $CONFIG_FILE --dry-run${RESET}"
echo -e "    ${BOLD}python dd_active_learning/dd_orchestrator.py --config $CONFIG_FILE${RESET}"
echo ""
echo "  Check progress any time:"
echo -e "    ${BOLD}python dd_active_learning/dd_status.py --config $CONFIG_FILE${RESET}"
echo ""