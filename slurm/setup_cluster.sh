#!/bin/bash
# =============================================================================
# setup_cluster.sh — Interactive one-time setup wizard for dd_prep
#
# Usage:
#   bash slurm/setup_cluster.sh
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

sacctmgr show associations user=$USER format=account%30 --noheader > /tmp/dd_accounts.txt 2>&1 &
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
# Accepts a single path, several space-separated paths, or a glob. A library
# pre-split across files does not need concatenating: the filter step streams
# them all into one filtered library.
#
# INPUT_YAML ends up holding the value written into the config, already in
# YAML form (either a quoted scalar or a flow sequence).
ask "Path to your input SMILES library:"
echo "    One file:      /project/lib/library.smi"
echo "    Several files: /project/lib/part1.smi /project/lib/part2.smi"
echo "    Or a glob:     /project/lib/part*.smi"
read -r INPUT_RAW

# Expand the answer into an array. Unquoted so both space separation and
# glob expansion happen. a glob that matches nothing stays literal, which
# the -f test below then rejects.
resolve_inputs() {
    local raw="$1"
    RESOLVED=()
    local entry
    # shellcheck disable=SC2086
    for entry in $raw; do
        if [ -f "$entry" ]; then
            RESOLVED+=("$entry")
        fi
    done
}

INPUT_YAML=""
while true; do
    resolve_inputs "$INPUT_RAW"
    N_INPUTS=${#RESOLVED[@]}

    if [ "$N_INPUTS" -eq 0 ]; then
        warn "No files matched: $INPUT_RAW"
        ask "Try again, or press Enter to skip and edit the config by hand:"
        read -r INPUT_RAW
        if [ -z "$INPUT_RAW" ]; then
            warn "Skipping. Set input_file in the config before running."
            INPUT_YAML='"/path/to/your/library.smi"'
            break
        fi
        continue
    fi

    TOTAL_SIZE=$(du -shc "${RESOLVED[@]}" 2>/dev/null | tail -1 | cut -f1)
    if [ "$N_INPUTS" -eq 1 ]; then
        success "Found 1 file (${TOTAL_SIZE}): ${RESOLVED[0]}"
        # Always quoted: an unquoted value containing a leading '*' is read
        # by YAML as an alias and makes the config unparseable.
        INPUT_YAML="\"${RESOLVED[0]}\""
    else
        success "Found $N_INPUTS files (${TOTAL_SIZE} total):"
        for entry in "${RESOLVED[@]}"; do
            echo "    $entry"
        done
        echo ""
        echo "  These will be merged into one filtered library by the filter step."
        ask "Use these $N_INPUTS files? (Y/n):"
        read -r CONFIRM
        if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
            ask "Enter the path, paths, or glob again:"
            read -r INPUT_RAW
            continue
        fi
        # YAML flow sequence, every element quoted.
        INPUT_YAML="["
        for entry in "${RESOLVED[@]}"; do
            INPUT_YAML="$INPUT_YAML\"$entry\", "
        done
        INPUT_YAML="${INPUT_YAML%, }]"
    fi
    break
done

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

# ── Make `module` available ───────────────────────────────────────────────────
# `module` is a shell function, and a non-interactive shell does not always
# inherit it. Source Lmod's init explicitly.
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh \
                /etc/profile.d/z00_lmod.sh \
                /usr/share/lmod/lmod/init/bash \
                "${LMOD_PKG:-/opt/lmod}/init/bash"; do
        # shellcheck disable=SC1090
        [ -r "$init" ] && source "$init" && break
    done
fi
if ! command -v module >/dev/null 2>&1; then
    error "The 'module' command is not available and could not be initialised."
    error "This script expects an Lmod/Environment-Modules cluster."
    exit 1
fi

# ── Leave any active virtualenv before touching modules ───────────────────────
if [ -n "${VIRTUAL_ENV:-}" ]; then
    warn "A virtualenv is active ($VIRTUAL_ENV); leaving it before loading modules."
    PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
    export PATH
    unset VIRTUAL_ENV
    unset PYTHONHOME
fi

# ── Load, checking each one ───────────────────────────────────────────────────
# `module load` of a missing module is not always fatal on its own, so each
# load is checked.
MODULE_ERR=/tmp/dd_prep_module_err.$$
require_module() {
    if ! module load "$@" > "$MODULE_ERR" 2>&1; then
        error "Failed to load module(s): $*"
        sed 's/^/    /' "$MODULE_ERR"
        echo ""
        error "Fix the module name for this cluster, then re-run setup."
        rm -f "$MODULE_ERR"
        exit 1
    fi
    # Lmod often reports problems on stderr while still exiting 0.
    if grep -qiE "error|cannot be loaded|not found|unable to locate" "$MODULE_ERR"; then
        error "Problem loading module(s): $*"
        sed 's/^/    /' "$MODULE_ERR"
        rm -f "$MODULE_ERR"
        exit 1
    fi
}

module purge
require_module StdEnv/2023
require_module python/3.11
require_module scipy-stack
require_module gcc rdkit
rm -f "$MODULE_ERR"
success "Modules loaded."

# ── Confirm the modules actually provide what they claim ──────────────────────
for pkg in rdkit pandas; do
    if ! python -c "import $pkg" 2>/tmp/dd_prep_imp_err.$$; then
        error "'$pkg' is not importable after loading modules."
        sed 's/^/    /' /tmp/dd_prep_imp_err.$$
        rm -f /tmp/dd_prep_imp_err.$$
        echo ""
        case "$pkg" in
            rdkit)  error "Check the rdkit module name: try 'module spider rdkit'." ;;
            pandas) error "pandas comes from scipy-stack: try 'module spider scipy-stack'." ;;
        esac
        exit 1
    fi
done
rm -f /tmp/dd_prep_imp_err.$$
success "Module-provided packages verified (rdkit, pandas)."

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

# pip install with live output (no --quiet) so progress is visible.
pip_run() {
    local label="$1"; shift
    info "$label"
    pip "$@" 2>&1 | tee "$VENV_DIR/.last_pip.log" | while IFS= read -r line; do
        printf "\r  ${BLUE}→${RESET}  %-70s" "$line"
    done
    local status=${PIPESTATUS[0]}
    printf "\r%-80s\r" " "
    if [ "$status" -ne 0 ]; then
        error "$label failed (pip exit $status)."
        echo ""
        echo "  Last 15 lines of pip output:"
        tail -15 "$VENV_DIR/.last_pip.log" | sed 's/^/    /'
        echo ""
        error "Setup aborted. Nothing downstream will work until this is fixed."
        exit 1
    fi
}

pip_run "Upgrading pip..."        install --upgrade pip
pip_run "Installing dependencies (pyyaml, tqdm)..." install pyyaml tqdm
pip_run "Installing dd_prep..."   install "$PROJECT_DIR" --force-reinstall --no-deps

# ── Verify ────────────────────────────────────────────────────────────────────
# We have two checks.
#
#   1. Imports, run from a directory that is not the repo. Python puts the
#      current directory on sys.path, so "import dd_prep" from the repo root
#      succeeds whether or not pip installed anything.
#   2. The dd-prep console script. The Slurm jobs invoke `dd-prep`, not
#      `import dd_prep`, so a missing entry point is what actually breaks a
#      run and it is invisible to an import check.
VERIFY_FAILED=false

DD_PREP_IMPORTS="rdkit, pandas, yaml, tqdm, dd_prep"
if (cd /tmp && python -c "import $DD_PREP_IMPORTS") 2>/dev/null; then
    success "Imports verified ($DD_PREP_IMPORTS)."
else
    error "Import verification failed."
    (cd /tmp && python -c "import $DD_PREP_IMPORTS") 2>&1 | tail -5 | sed 's/^/    /'
    VERIFY_FAILED=true
fi

if command -v dd-prep >/dev/null 2>&1; then
    success "Console script found: $(command -v dd-prep)"
else
    error "The 'dd-prep' command is not on PATH after installation."
    echo "    The SLURM scripts call 'dd-prep' directly, so jobs would fail"
    echo "    at startup with 'command not found' (exit code 127)."
    echo "    Workaround: use 'python -m dd_prep.cli' instead."
    VERIFY_FAILED=true
fi

if [ "$VERIFY_FAILED" = true ]; then
    error "Setup incomplete. Resolve the above before submitting jobs."
    exit 1
fi

# ── Step 5: OpenEye ───────────────────────────────────────────────────────────
echo ""
info "OpenEye configuration"
echo "  OpenEye is required for the flipper and tautomers steps."
echo "  Press Enter to skip if you don't have OpenEye access."
echo ""

ask "Auto-detect OpenEye installation? This searches the filesystem and may take 1-2 minutes. (Y/n):"
read -r OE_AUTO
OE_BIN=""
OE_LIC=""

if [[ ! "$OE_AUTO" =~ ^[Nn]$ ]]; then
    # Auto-detect
    info "Searching for OpenEye binaries..."
    find /project /opt /software 2>/dev/null         -name "flipper"         -not -path "*/arch/*"         -not -path "*/omega/*"         > /tmp/dd_oe_bin.txt 2>/dev/null &
    spinner $! "Searching filesystem for OpenEye (Ctrl+C to cancel and enter manually)..."
    wait $! || true
    OE_BIN=$(head -1 /tmp/dd_oe_bin.txt | xargs -I{} dirname {} 2>/dev/null || true)

    if [ -n "$OE_BIN" ]; then
        success "Found OpenEye binaries: $OE_BIN"
    else
        warn "Could not find OpenEye binaries automatically."
        ask "Enter path to OpenEye bin directory manually (or press Enter to skip):"
        read -r OE_BIN_INPUT
        OE_BIN="${OE_BIN_INPUT:-}"
    fi

    info "Searching for OpenEye licence..."
    find /project /opt /home/$USER 2>/dev/null         -name "oe_license.txt"         -not -path "*/arch/*"         -not -name "*.bak*"         > /tmp/dd_oe_lic.txt 2>/dev/null &
    spinner $! "Searching for licence file (Ctrl+C to cancel and enter manually)..."
    wait $! || true
    OE_LIC=$(head -1 /tmp/dd_oe_lic.txt || true)

    if [ -n "$OE_LIC" ]; then
        success "Found OpenEye licence: $OE_LIC"
    else
        warn "Could not find licence file automatically."
        ask "Enter path to oe_license.txt manually (or press Enter to skip):"
        read -r OE_LIC_INPUT
        OE_LIC="${OE_LIC_INPUT:-}"
    fi
else
    # Manual entry
    ask "Path to OpenEye bin directory (contains flipper, tautomers):"
    read -r OE_BIN_INPUT
    OE_BIN="${OE_BIN_INPUT:-}"

    ask "Path to oe_license.txt:"
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
# Skip this script. Bash reads a script incrementally by byte offset, so
# rewriting it while it is still executing can shift those offsets and make
# bash resume mid-token -- producing a syntax error on a line that is
# perfectly valid, somewhere after this loop. setup_cluster.sh has no
# #SBATCH --account line to update anyway.
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"

for f in "$SCRIPT_DIR"/*.sh; do
    [ "$(basename "$f")" = "$SELF_NAME" ] && continue
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
#
# Usage:  source .dd_prep_env
#
# Loads the modules and activates the virtualenv in that order.

# Make 'module' available in non-interactive shells.
if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/z00_lmod.sh \\
                /usr/share/lmod/lmod/init/bash; do
        [ -r "\$init" ] && source "\$init" && break
    done
fi

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load gcc rdkit

export DD_PREP_VENV="$VENV_DIR"
export DD_PREP_PROJECT="$PROJECT_DIR"
export DD_PREP_CONFIG="$PROJECT_DIR/my_run.yaml"

source "\$DD_PREP_VENV/bin/activate"
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

input_file: $INPUT_YAML
work_dir:   "$WORK_DIR"

n_parallel: 4
resume: true

# Delete each stage's input once its output is written (filtered file, chunks,
# _isom). Recommended for large libraries on quota-limited /project space.
# A cleaned stage can't be re-run from scratch (output-based resume still works).
cleanup_intermediates: false

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

  # Max tetrahedral stereocentres per molecule, counting both declared
  # (@ / @@) and unspecified centres. null = no limit.
  # Flipper enumerates every unspecified centre, so a molecule with n
  # undeclared centres becomes 2^n isomers.
  chiral_centers_max: null

  # Every threshold above accepts null to switch it off, and the descriptor
  # is then not computed. For a single-property pass (e.g. stereocentres
  # only) set all the others to null.

  # RDKit worker processes — the filter throughput lever.
  # Match --cpus-per-task in slurm/01_filter_split.sh.
  n_workers: 32

split:
  chunk_size: 10000000

flipper:
  enabled: $FLIPPER_ENABLED
  warts: true
  enum_nitrogen: false

tautomer:
  enabled: $TAUTOMER_ENABLED
  max_to_return: 1
  # KEEP FALSE.
  ch3: false
  warts: false

organize:
  enabled: true
  # hardlink (no extra disk, keeps intermediate) | move | copy
  mode: hardlink

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