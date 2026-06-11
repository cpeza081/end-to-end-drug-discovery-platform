#!/bin/bash
# =============================================================================
# setup_cluster.sh, an interactive one-time setup wizard for dd_prep
# 
# Usage: 
#   bash slurm/setup_cluster.sh
# 
# Press Ctrl+C at any time to cancel cleanly
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
# Kills any background jobs (spinner, find, pip) before exiting cleanly.
cleanup() {
    echo ""
    echo ""
    warn "Setup cancelled by user."
    # Kill all child processes started by this script
    kill $(jobs -p) 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── Spinner ───────────────────────────────────────────────────────────────────
# Shows an animated spinner while a background process runs.
# Usage:
#   some_slow_command &
#   spinner $! "Doing something..."
#   wait $!    # check exit code
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
    printf "\r%-60s\r" " "   # clear the spinner line
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   Deep Docking Library Preparation — Cluster Setup Wizard${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Press Ctrl+C at any time to cancel."
echo "  Press Enter to accept defaults shown in [brackets]."
echo ""

# ── Step 1: SLURM account ─────────────────────────────────────────────────────
info "Detecting your SLURM accounts..."
 
sacctmgr show associations user=$USER format=account --noheader > /tmp/dd_accounts.txt 2>&1 &
spinner $! "Querying SLURM..."
wait $!

ACCOUNTS=$(cat /tmp/dd_accounts.txt | tr -d ' ' | grep -v "^$" | sort -u)
 
if [ -z "$ACCOUNTS" ]; then
    error "No SLURM accounts found. Contact your system administrator."
    exit 1
fi

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
success "Using account: $SLURM_ACCOUNT"

# ── Step 2: Input SMILES library ──────────────────────────────────────────────
ask "Full path to your input SMILES library (.smi file):"
read -r INPUT_FILE
 
if [ -f "$INPUT_FILE" ]; then
    N_MOLS=$(wc -l < "$INPUT_FILE")
    success "Found library (~$N_MOLS lines)."
else
    warn "File not found: $INPUT_FILE"
    warn "You can update input_file in the config later."
    INPUT_FILE="/path/to/your/library.smi"
fi

# ── Step 3: Output directory ──────────────────────────────────────────────────
DEFAULT_WORK_DIR="$SCRATCH/dd_prep_output"
ask "Where should pipeline outputs go? [$DEFAULT_WORK_DIR]:"
read -r WORK_DIR_INPUT
WORK_DIR="${WORK_DIR_INPUT:-$DEFAULT_WORK_DIR}"
success "Output directory: $WORK_DIR"

# ── Step 4: Virtual environment ───────────────────────────────────────────────
DEFAULT_VENV="$SCRATCH/dd_prep_venv"
ask "Where should the virtual environment be created? [$DEFAULT_VENV]:"
read -r VENV_INPUT
VENV_DIR="${VENV_INPUT:-$DEFAULT_VENV}"

echo ""
info "Loading modules..."
module purge
module load StdEnv/2023
module load python/3.11
module load gcc rdkit
success "Modules loaded."

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR"
    ask "Recreate it from scratch? (y/N):"
    read -r RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        info "Removed existing environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python -m venv --system-site-packages "$VENV_DIR" &
    spinner $! "Creating virtual environment..."
    wait $!
    success "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"
 
# pip install with live output (no --quiet) so progress is visible
info "Upgrading pip..."
pip install --upgrade pip 2>&1 | while IFS= read -r line; do
    printf "\r  ${BLUE}→${RESET}  %-70s" "$line"
done
printf "\r%-80s\r" " "
success "pip upgraded."

info "Installing dependencies (pyyaml, pandas, tqdm)..."
pip install pyyaml pandas tqdm 2>&1 | while IFS= read -r line; do
    printf "\r  ${BLUE}→${RESET}  %-70s" "$line"
done
printf "\r%-80s\r" " "
success "Dependencies installed."

info "Installing dd_prep..."
pip install "$PROJECT_DIR" --force-reinstall 2>&1 | while IFS= read -r line; do
    printf "\r  ${BLUE}→${RESET}  %-70s" "$line"
done
printf "\r%-80s\r" " "

# Verify
python -c "import rdkit, dd_prep, pandas, yaml" 2>/dev/null \
    && success "All imports verified (rdkit, dd_prep, pandas, yaml)." \
    || { error "Import verification failed. Check output above."; exit 1; }


# ── Step 5: OpenEye ───────────────────────────────────────────────────────────
echo ""
info "Searching for OpenEye binaries..."
 
# Run find in background with spinner
find /project /opt /software 2>/dev/null \
    -name "flipper" \
    -not -path "*/arch/*" \
    -not -path "*/omega/*" \
    > /tmp/dd_oe_bin.txt 2>/dev/null &
spinner $! "Searching filesystem for OpenEye..."
wait $! || true

OE_BIN=$(head -1 /tmp/dd_oe_bin.txt | xargs -I{} dirname {} 2>/dev/null || true)
 
if [ -n "$OE_BIN" ]; then
    success "Found OpenEye binaries: $OE_BIN"
else
    warn "Could not auto-detect OpenEye binaries."
    ask "Enter the full path to your OpenEye bin directory (or press Enter to skip):"
    read -r OE_BIN_INPUT
    OE_BIN="${OE_BIN_INPUT:-}"
fi

info "Searching for OpenEye licence..."
find /project /opt /home/$USER 2>/dev/null \
    -maxdepth 6 \
    -name "oe_license.txt" \
    -not -path "*/arch/*" \
    -not -name "*.bak*" \
    > /tmp/dd_oe_lic.txt 2>/dev/null &
spinner $! "Searching for licence file..."
wait $! || true

OE_LIC=$(head -1 /tmp/dd_oe_lic.txt || true)
 
if [ -n "$OE_LIC" ]; then
    success "Found OpenEye licence: $OE_LIC"
else
    warn "Could not auto-detect OpenEye licence."
    ask "Enter the full path to oe_license.txt (or press Enter to skip):"
    read -r OE_LIC_INPUT
    OE_LIC="${OE_LIC_INPUT:-}"
fi

OE_WORKS=false
if [ -n "$OE_BIN" ] && [ -n "$OE_LIC" ] && [ -f "$OE_LIC" ]; then
    export OE_LICENSE="$OE_LIC"
    export PATH="$OE_BIN:$PATH"
    command -v flipper &>/dev/null \
        && { success "OpenEye verified (flipper found)."; OE_WORKS=true; } \
        || warn "flipper not found at $OE_BIN"
fi

[ "$OE_WORKS" = false ] && warn "Flipper/tautomers will be disabled in the config. Enable them later once OpenEye is set up."
 
# ── Step 6: Update SLURM scripts ─────────────────────────────────────────────
info "Updating SLURM scripts..."
for f in "$SCRIPT_DIR"/*.sh; do
    sed -i 's/\r//' "$f"
    sed -i "s|#SBATCH --account=.*|#SBATCH --account=$SLURM_ACCOUNT|g" "$f"
    [ -n "$OE_LIC" ] && sed -i "s|export OE_LICENSE=.*|export OE_LICENSE=\"$OE_LIC\"|g" "$f"
    if [ -n "$OE_BIN" ] && ! grep -q "OE_BIN_PATH" "$f"; then
        sed -i "s|export OE_LICENSE=|export PATH=\"$OE_BIN:\$PATH\"\nexport OE_LICENSE=|g" "$f"
    fi
done
success "All SLURM scripts updated."

# ── Step 7: Write .dd_prep_env ────────────────────────────────────────────────
cat > "$PROJECT_DIR/.dd_prep_env" << ENVEOF
# Auto-generated by setup_cluster.sh on $(date)
export DD_PREP_VENV="$VENV_DIR"
export DD_PREP_PROJECT="$PROJECT_DIR"
ENVEOF
success "Environment saved to .dd_prep_env"
 
# ── Step 8: Generate config ───────────────────────────────────────────────────
CONFIG_FILE="$PROJECT_DIR/my_run.yaml"
 
if [ -f "$CONFIG_FILE" ]; then
    ask "Config already exists at $CONFIG_FILE. Overwrite? (y/N):"
    read -r OW
    [[ "$OW" =~ ^[Yy]$ ]] || CONFIG_FILE="$PROJECT_DIR/my_run_$(date +%Y%m%d_%H%M%S).yaml"
fi
 
FLIPPER_ENABLED=$( [ "$OE_WORKS" = true ] && echo "true" || echo "false" )
TAUTOMER_ENABLED=$( [ "$OE_WORKS" = true ] && echo "true" || echo "false" )
 
cat > "$CONFIG_FILE" << YAML
# dd_prep configuration — generated $(date)
 
input_file: $INPUT_FILE
work_dir:   $WORK_DIR
 
n_parallel: 4
resume: true
 
filter:
  enabled: true
  slogp_min: 1.0
  slogp_max: 3.5
  rot_bonds_max: 6
  mw_min: 300.0
  mw_max: 450.0
  fsp3_min: 0.25
  aro_rings_min: 1
  aro_rings_max: 2
  aliph_rings_max: 3
  total_rings_min: 3
  total_rings_max: 4
  formal_charge: 0
 
split:
  chunk_size: 10000000
 
flipper:
  enabled: $FLIPPER_ENABLED
  warts: true
  enum_nitrogen: false
 
tautomer:
  enabled: $TAUTOMER_ENABLED
  max_to_return: 1
  ch3: false
 
organize:
  enabled: true
 
fingerprint:
  enabled: true
  radius: 2
  n_bits: 1024
  n_workers: 16
 
omega:
  enabled: false
YAML
 
success "Config written to: $CONFIG_FILE"
 
# ── Step 9: Validate ──────────────────────────────────────────────────────────
echo ""
info "Running validation..."
dd-prep --config "$CONFIG_FILE" --validate-only 2>&1 \
    | grep -v "No chunk\|No input\|No processed\|No prepared" \
    || true
success "Setup complete."
 
# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}   All done!${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Account   : $SLURM_ACCOUNT"
echo "  Config    : $CONFIG_FILE"
echo "  Venv      : $VENV_DIR"
echo "  Outputs   : $WORK_DIR"
[ -n "$OE_LIC" ] && echo "  OE Lic    : $OE_LIC"
echo ""
echo -e "  Run the pipeline:     ${BOLD}bash slurm/submit_pipeline.sh${RESET}"
echo -e "  Check progress:       ${BOLD}bash slurm/status.sh${RESET}"
echo ""