#!/bin/bash
# =============================================================================
# setup_active_learning.sh
#
# Interactive one-time setup wizard for dd_active_learning.
#
# Usage:
#   bash dd_active_learning/setup_active_learning.sh
#
# What it does:
#   * finds (or clones) the DD_protocol scripts
#   * picks the docking engine (Gnina or AutoDock-GPU) and binding-site strategy
#   * records the cluster modules that provide the docking tools
#   * creates the conda/mamba software environment (from DD_protocol's
#     environment.yml, plus Meeko) if you don't already have one
#   * collects the Deep Docking parameters (with explanations from the DD paper)
#   * writes a ready-to-run campaign.yaml and validates it
#
# Press Ctrl+C at any time to cancel cleanly.
# =============================================================================

# --- Colours ----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
ask()     { echo -e "\n${BOLD}$*${RESET}"; }
note()    { echo -e "        ${*}"; }

# --- Ctrl+C / cancellation --------------------------------------------------
cleanup() {
    echo ""; echo ""
    warn "Setup cancelled by user."
    kill "$(jobs -p)" 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- Spinner ----------------------------------------------------------------
spinner() {
    local pid=$1 label="$2"
    local frames=('|' '/' '-' '\')
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${BLUE}%s${RESET}  %s" "${frames[$i]}" "$label"
        i=$(( (i + 1) % ${#frames[@]} ))
        sleep 0.1
    done
    printf "\r%-70s\r" " "
}

# Bounded filesystem search: never runs longer than SEARCH_TIMEOUT seconds, so
# the wizard can never hang on a slow/huge filesystem.  Result -> DETECT_OUT.
SEARCH_TIMEOUT=45
DETECT_OUT=""
detect_path() {
    # detect_path "<spinner label>" <find-args...>
    local label="$1"; shift
    local tmp; tmp=$(mktemp)
    ( timeout "$SEARCH_TIMEOUT" find "$@" -print 2>/dev/null | head -1 > "$tmp" ) &
    spinner $! "$label (up to ${SEARCH_TIMEOUT}s; Ctrl+C to cancel)..."
    wait $! 2>/dev/null || true
    DETECT_OUT=$(head -1 "$tmp" 2>/dev/null)
    rm -f "$tmp"
}

# prompt with a default -> REPLY_VAL
REPLY_VAL=""
prompt_default() {
    local p="$1" d="$2"
    ask "$p [$d]:"
    read -r REPLY_VAL
    REPLY_VAL="${REPLY_VAL:-$d}"
}

yes_no() {   # yes_no "question" "Y"|"N"  -> returns 0 for yes
    local q="$1" def="${2:-Y}" ans
    if [ "$def" = "Y" ]; then ask "$q (Y/n):"; else ask "$q (y/N):"; fi
    read -r ans
    ans="${ans:-$def}"
    [[ "$ans" =~ ^[Yy]$ ]]
}

# Install a startup shim into the python env that restores the
# numpy aliases (np.bool, np.int, ...) removed in numpy>=1.24, so the unmodified
# DD_protocol code keeps working on the modern numpy that TF 2.15 requires.
# We use a .pth file: a .pth "import ..." line runs at
# every interpreter startup.
_install_numpy_shim() {
    local sp
    sp=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null) || return 0
    [ -z "$sp" ] && return 0
    printf '%s\n' "import numpy as _np, builtins as _b; ('bool' in _np.__dict__) or setattr(_np,'bool',_np.bool_); ('int' in _np.__dict__) or setattr(_np,'int',_b.int); ('float' in _np.__dict__) or setattr(_np,'float',_np.float64); ('object' in _np.__dict__) or setattr(_np,'object',_b.object); ('str' in _np.__dict__) or setattr(_np,'str',_b.str); ('complex' in _np.__dict__) or setattr(_np,'complex',_np.complex128)" \
        > "$sp/dd_numpy_compat.pth"
    success "Installed numpy-compat shim (restores np.bool for the DD scripts)."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   Deep Docking Active Learning Setup Wizard${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Press Ctrl+C at any time to cancel."
echo "  Press Enter to accept defaults shown in [brackets]."
echo ""

# =============================================================================
# Step 1: SLURM account
# =============================================================================
info "Detecting your SLURM accounts (a few seconds)..."
ACCOUNTS=""
if command -v sacctmgr &>/dev/null; then
    ( timeout 20 sacctmgr show associations user="$USER" format=account%30 --noheader \
        > /tmp/dd_al_accounts.txt 2>/dev/null ) &
    spinner $! "Querying SLURM..."
    wait $! 2>/dev/null || true
    ACCOUNTS=$(tr -d ' ' < /tmp/dd_al_accounts.txt | grep -v "^$" | sort -u)
else
    warn "sacctmgr not found; you can fill in the account later."
fi

if [ -z "$ACCOUNTS" ]; then
    prompt_default "Enter your billing account name" "CHANGE_ME"
    SLURM_ACCOUNT="$REPLY_VAL"
else
    echo ""; echo "  Available accounts:"
    i=1; declare -a ACCOUNT_LIST
    while IFS= read -r acc; do echo "    $i) $acc"; ACCOUNT_LIST[$i]="$acc"; ((i++)); done <<< "$ACCOUNTS"
    prompt_default "Which account should jobs be billed to? Enter number" "1"
    SLURM_ACCOUNT="${ACCOUNT_LIST[$REPLY_VAL]:-CHANGE_ME}"
fi
success "Using account: $SLURM_ACCOUNT"

# =============================================================================
# Step 2: Prepared library (link a finished dd_prep run)
# =============================================================================
ask "How will the prepared library be provided?"
echo "    1) Link an existing finished dd_prep run (recommended)"
echo "    2) I'll set library paths manually later"
read -r LIBRARY_CHOICE; LIBRARY_CHOICE="${LIBRARY_CHOICE:-1}"

SMILES_DIR="\$SCRATCH/library_prepared"
FP_DIR="\$SCRATCH/library_prepared_fp"
if [ "$LIBRARY_CHOICE" = "1" ]; then
    ask "Path to the dd_prep work_dir (the folder containing library_prepared/ and library_prepared_fp/):"
    read -r PREP_WORK_DIR
    if [ -f "$PREP_WORK_DIR" ]; then
        PREP_WORK_DIR=$(python3 -c "import yaml,sys;print((yaml.safe_load(open('$PREP_WORK_DIR')) or {}).get('work_dir',''))" 2>/dev/null)
    fi
    if [ -d "$PREP_WORK_DIR/library_prepared" ] && [ -d "$PREP_WORK_DIR/library_prepared_fp" ]; then
        SMILES_DIR="$(cd "$PREP_WORK_DIR/library_prepared" && pwd)"
        FP_DIR="$(cd "$PREP_WORK_DIR/library_prepared_fp" && pwd)"
        success "Found prepared library under $PREP_WORK_DIR"
    else
        warn "Could not find library_prepared/ + library_prepared_fp/ there."
        note "Link later: python dd_link.py --prep-work-dir <work_dir> --campaign <config>"
    fi
fi

# =============================================================================
# Step 3: Campaign identity
# =============================================================================
prompt_default "Campaign name (used for directories and job names)" "my_target_dd"
CAMPAIGN_NAME="$REPLY_VAL"
prompt_default "Where should campaign outputs go?" "\$SCRATCH/dd_campaigns/$CAMPAIGN_NAME"
CAMPAIGN_DIR_RAW="$REPLY_VAL"
CAMPAIGN_DIR_EXPANDED=$(eval echo "$CAMPAIGN_DIR_RAW")
success "Campaign output directory: $CAMPAIGN_DIR_RAW"

# =============================================================================
# Step 4: DD_protocol scripts
# =============================================================================
REQUIRED_SCRIPTS=(
    "scripts_1/molecular_file_count_updated.py" "scripts_1/sampling.py"
    "scripts_1/sanity_check.py" "scripts_1/extracting_morgan.py"
    "scripts_1/extracting_smiles.py" "scripts_2/extract_labels.py"
    "scripts_2/simple_job_models_manual.py"
    "scripts_2/hyperparameter_result_evaluation.py"
    "scripts_2/simple_job_predictions_manual.py" "utilities/final_extraction.py"
)
validate_dd_protocol_dir() {
    local c="$1" s
    for s in "${REQUIRED_SCRIPTS[@]}"; do [ -f "$c/$s" ] || return 1; done
    return 0
}

DD_PROTOCOL_DIR=""
if yes_no "Do you already have a DD_protocol checkout on this system?" "N"; then
    if yes_no "Auto-detect it? (filesystem search, bounded to ${SEARCH_TIMEOUT}s)" "Y"; then
        detect_path "Searching for DD_protocol" \
            "$HOME" "${SCRATCH:-/scratch}" /project 2>/dev/null -maxdepth 6 -type d -name "DD_protocol"
        if [ -n "$DETECT_OUT" ] && validate_dd_protocol_dir "$DETECT_OUT"; then
            DD_PROTOCOL_DIR="$DETECT_OUT"
            success "Found valid DD_protocol: $DD_PROTOCOL_DIR"
        else
            warn "Auto-detect did not find a valid checkout."
        fi
    fi
    if [ -z "$DD_PROTOCOL_DIR" ]; then
        ask "Enter the path to your DD_protocol checkout:"
        read -r DD_PROTOCOL_DIR
    fi
fi

if [ -z "$DD_PROTOCOL_DIR" ] || ! validate_dd_protocol_dir "$DD_PROTOCOL_DIR"; then
    [ -n "$DD_PROTOCOL_DIR" ] && warn "That path is missing required scripts."
    if yes_no "Clone DD_protocol now from github.com/jamesgleave/DD_protocol?" "Y"; then
        prompt_default "Clone destination" "\$SCRATCH/DD_protocol"
        CLONE_DIR=$(eval echo "$REPLY_VAL")
        git clone https://github.com/jamesgleave/DD_protocol "$CLONE_DIR" &
        spinner $! "Cloning DD_protocol..."
        if wait $! && validate_dd_protocol_dir "$CLONE_DIR"; then
            DD_PROTOCOL_DIR="$CLONE_DIR"; success "Cloned: $CLONE_DIR"
        else
            error "Clone failed or scripts missing; set dd_protocol_dir manually later."
            DD_PROTOCOL_DIR="$CLONE_DIR"
        fi
    else
        DD_PROTOCOL_DIR="\$SCRATCH/DD_protocol"
        warn "Set env.dd_protocol_dir in campaign.yaml before launching."
    fi
fi

# =============================================================================
# Step 5: Docking engine
# =============================================================================
ask "Which docking engine will this campaign use?"
echo "    1) Gnina        (CNN-rescored docking; one multi-molecule SDF/chunk)"
echo "    2) AutoDock-GPU (grid-map docking, one PDBQT per ligand)"
read -r ENGINE_CHOICE; ENGINE_CHOICE="${ENGINE_CHOICE:-1}"
ADD=""
if [ "$ENGINE_CHOICE" = "2" ]; then
    DOCK_PROGRAM="AUTODOCK_GPU"; SCORE_KEYWORD="ADGPU_score"
    DEFAULT_MODULES="autodock-gpu autodock"
    ADD=1   # AutoDock-GPU also needs grid maps built by dd_receptor_prep
else
    DOCK_PROGRAM="GNINA"; SCORE_KEYWORD="minimizedAffinity"
    DEFAULT_MODULES="gnina"
fi
success "Engine: $DOCK_PROGRAM  (score field: $SCORE_KEYWORD)"

# --- Receptor + binding site ---
ask "Path to the receptor structure (PDB):"
read -r RECEPTOR_FILE
RECEPTOR_FILE="${RECEPTOR_FILE:-\$SCRATCH/receptor/receptor.pdb}"
RECEPTOR_DIR=$(dirname "$RECEPTOR_FILE")
BOX_JSON="$RECEPTOR_DIR/receptor_box.json"
MAPS_FLD="$RECEPTOR_DIR/receptor.maps.fld"

echo ""
info "Binding-site strategy (Deep Docking docks the whole library into one site):"
echo "    1) reference_ligand  - box from a known ligand in the pocket (most accurate)"
echo "    2) manual            - you type the box center + size"
echo "    3) p2rank            - predict the pocket from the protein (no ligand needed)"
read -r SITE_CHOICE; SITE_CHOICE="${SITE_CHOICE:-1}"
SITE_YAML=""
if [ "$SITE_CHOICE" = "2" ]; then
    SITE_METHOD="manual"
    prompt_default "Box center x y z (space-separated)" "0 0 0"; CENTER="$REPLY_VAL"
    prompt_default "Box size   x y z (Angstrom)" "22 22 22"; SIZE="$REPLY_VAL"
    read -r CX CY CZ <<< "$CENTER"; read -r SX SY SZ <<< "$SIZE"
    SITE_YAML=$(printf '    method: "manual"\n    center: [%s, %s, %s]\n    size: [%s, %s, %s]' "$CX" "$CY" "$CZ" "$SX" "$SY" "$SZ")
elif [ "$SITE_CHOICE" = "3" ]; then
    SITE_METHOD="p2rank"
    prompt_default "P2Rank launcher (module/exec name)" "prank"; P2RANK_EXEC="$REPLY_VAL"
    prompt_default "Which predicted pocket to target (1 = top)" "1"; POCKET_RANK="$REPLY_VAL"
    prompt_default "Box size x y z (Angstrom)" "24 24 24"; P2SIZE="$REPLY_VAL"
    read -r PX PY PZ <<< "$P2SIZE"
    SITE_YAML=$(printf '    method: "p2rank"\n    p2rank_exec: "%s"\n    pocket_rank: %s\n    box_size: [%s, %s, %s]' "$P2RANK_EXEC" "$POCKET_RANK" "$PX" "$PY" "$PZ")
else
    SITE_METHOD="reference_ligand"
    ask "Path to the reference ligand (SDF/MOL/MOL2/PDB) in the target pocket:"
    read -r REF_LIGAND
    REF_LIGAND="${REF_LIGAND:-\$SCRATCH/receptor/ref_ligand.sdf}"
    prompt_default "Padding around the ligand (Angstrom)" "4.0"; PADDING="$REPLY_VAL"
    SITE_YAML=$(printf '    method: "reference_ligand"\n    reference_ligand: "%s"\n    padding: %s' "$REF_LIGAND" "$PADDING")
fi

# --- Engine-specific defaults ---
if [ "$DOCK_PROGRAM" = "GNINA" ]; then
    GNINA_CNN="rescore"; EXHAUST="8"
else
    AUTODOCK_BIN="autodock_gpu_128wi"; AUTODOCK_NRUN="10"
fi

# =============================================================================
# Step 6: Docking-engine modules - resolved automatically, no prompts
# =============================================================================
# You always use Gnina or AutoDock-GPU, so the module chain is deterministic.
# The wizard resolves it from Lmod (`module spider` reports the exact
# prerequisites + versioned name) with zero user input.  Hardcoded fallbacks
# are used if the module system can't be queried from here.
if [ "$DOCK_PROGRAM" = "GNINA" ]; then
    ENGINE_MOD="gnina"
    FALLBACK_MODS="StdEnv/2023 gcc/12.3 cuda/12.2 gnina/1.3.1"
else
    ENGINE_MOD="autodock-gpu"
    FALLBACK_MODS="StdEnv/2023 gcc/12.3 cuda/12.2 autodock-gpu/1.6"
fi

DOCK_MODS="$FALLBACK_MODS"
if type module &>/dev/null; then
    SPIDER=$(module spider "$ENGINE_MOD" 2>&1)
    # exact versioned name (e.g. gnina/1.3.1)
    FULL=$(printf '%s\n' "$SPIDER" | grep -oE "${ENGINE_MOD}/[0-9][^ :]*" | head -1)
    # the line right after "You will need to load ..." lists the prerequisites
    PREREQ=$(printf '%s\n' "$SPIDER" | grep -A1 "You will need to load" | tail -1 | xargs)
    # Only trust the resolved chain if we got both the versioned name and its
    # prerequisites, otherwise keep the hardcoded fallback (which always has the
    # prereqs).
    if [ -n "$FULL" ] && [ -n "$PREREQ" ]; then
        DOCK_MODS=$(echo "$PREREQ $FULL" | xargs)
        success "Resolved '$ENGINE_MOD' module chain automatically."
    else
        warn "Could not fully resolve '$ENGINE_MOD' prerequisites. Using known defaults."
    fi
fi
info "Docking modules: $DOCK_MODS"
read -ra MODULE_ARR <<< "$DOCK_MODS"
# MODULES_YAML is finalized just before the config is written (a venv also needs
# its python module loaded first).

# =============================================================================
# Step 7: Software environment - the wizard BUILDS it (conda OR a pip venv)
# =============================================================================
# Prefer conda/mamba (from dd_environment.yml).  If neither is on PATH (common
# on clusters that use `module load python` + pip instead of conda (e.g. the
# Digital Research Alliance of Canada)) build a Python virtualenv from
# dd_requirements.txt.  Either way the environment is CREATED here.
# ENV_ACTIVATE is the command the generated job scripts use to enter it.
ENV_YML="$SCRIPT_DIR/dd_environment.yml"
REQ_TXT="$SCRIPT_DIR/dd_requirements.txt"
REQUIRED_IMPORTS="import tensorflow, rdkit, meeko, sklearn, pandas, numpy"
ENV_ACTIVATE=""
CONDA_ENV=""

PKG_MGR=""
command -v mamba &>/dev/null && PKG_MGR="mamba"
[ -z "$PKG_MGR" ] && command -v conda &>/dev/null && PKG_MGR="conda"

if [ -n "$PKG_MGR" ] && [ -f "$ENV_YML" ]; then
    # -------------------- conda / mamba path --------------------
    DEFAULT_ENV=$(awk '/^name:/{print $2; exit}' "$ENV_YML" 2>/dev/null)
    prompt_default "Conda environment name" "${DEFAULT_ENV:-dd-env}"
    CONDA_ENV="$REPLY_VAL"
    # \$( is escaped so it stays literal in the config and runs at job time.
    ENV_ACTIVATE="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate \"$CONDA_ENV\""
    conda_env_exists() { conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}[[:space:]]"; }
    conda_ok() { conda run -n "$CONDA_ENV" python -c "$REQUIRED_IMPORTS" 2>/dev/null; }
    if conda_env_exists; then
        success "Conda env '$CONDA_ENV' already exists (using $PKG_MGR)."
        conda_ok && success "It has the required packages." \
            || { yes_no "Update it from dd_environment.yml?" "Y" \
                 && "$PKG_MGR" env update -n "$CONDA_ENV" -f "$ENV_YML"; }
    else
        info "Creating conda env '$CONDA_ENV' from dd_environment.yml (~5-15 min)..."
        if "$PKG_MGR" env create -n "$CONDA_ENV" -f "$ENV_YML" && conda_ok; then
            success "Environment created and verified."
        else
            warn "Env build finished with warnings - check the log above."
        fi
    fi
else
    # -------------------- python virtualenv path --------------------
    [ -z "$PKG_MGR" ] && info "No conda/mamba found - building a Python virtualenv with pip."
    # Load the python module the venv symlinks to. For RDKit, Meeko needs rdkit.Chem.rdDetermineBonds,
    # which some Alliance rdkit builds omit, so probe the module versions
    # (newest first) for one that provides it. If found we use that
    # module, otherwise RDKit is installed from PyPI below.
    PY_MODULE=""; RDKIT_MODULE=""
    if type module &>/dev/null; then
        info "Loading python + rdkit modules (gcc prerequisite)..."
        module load gcc 2>/dev/null || true
        module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null \
            || module load python 2>/dev/null || true
        PY_MODULE=$(module -t list 2>&1 | grep -iE '^python/' | head -1)
        RDKIT_FULL=$(module spider rdkit 2>&1 | grep -oE "rdkit/[0-9][^ :]*" | head -1)
        [ -n "$RDKIT_FULL" ] && module load "$RDKIT_FULL" 2>/dev/null
        RDKIT_MODULE=$(module -t list 2>&1 | grep -iE '^rdkit/' | head -1)
        [ -n "$RDKIT_MODULE" ] && success "RDKit will come from module: $RDKIT_MODULE"
    fi
    PYTHON=$(command -v python3 || command -v python || true)
    if [ -z "$PYTHON" ]; then
        error "No python3 on PATH. Run 'module load python', then re-run the wizard."
    elif [ ! -f "$REQ_TXT" ]; then
        error "Bundled requirements not found: $REQ_TXT"
    else
        prompt_default "Path for the new Python virtual environment" "\$SCRATCH/dd_venv"
        VENV_RAW="$REPLY_VAL"; VENV_DIR=$(eval echo "$VENV_RAW")
        ENV_ACTIVATE="source \"$VENV_RAW/bin/activate\""

        # If a venv already exists, offer to delete + rebuild it (confirmed, and
        # only if the path really looks like a venv and isn't something dangerous).
        if [ -f "$VENV_DIR/bin/activate" ]; then
            warn "A virtual environment already exists at: $VENV_DIR"
            if yes_no "Delete it and rebuild from scratch?" "N"; then
                case "$VENV_DIR" in
                    ""|"/"|"$HOME"|"${SCRATCH:-/scratch}")
                        error "Refusing to delete '$VENV_DIR' (unsafe path). Skipping." ;;
                    *)
                        if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
                            info "Removing $VENV_DIR ..."; rm -rf -- "$VENV_DIR"
                        else
                            warn "$VENV_DIR has no pyvenv.cfg (not a venv?); NOT deleting."
                        fi ;;
                esac
            fi
        fi

        if [ ! -f "$VENV_DIR/bin/activate" ]; then
            info "Creating virtualenv at $VENV_DIR ..."
            "$PYTHON" -m venv "$VENV_DIR"
        fi
        if [ -f "$VENV_DIR/bin/activate" ]; then
            # shellcheck disable=SC1090
            source "$VENV_DIR/bin/activate"
            if ! python -c "$REQUIRED_IMPORTS" 2>/dev/null; then
                info "Installing core packages from dd_requirements.txt (~5-15 min)..."
                pip install --upgrade pip >/dev/null 2>&1 || true
                pip install -r "$REQ_TXT"
                # Meeko + gemmi from PyPI. RDKit is installed from PyPI only if no
                # module version provided rdDetermineBonds above, otherwise RDKit
                # stays the cluster module.
                if ! python -c "import meeko" 2>/dev/null; then
                    info "Installing Meeko (+gemmi) from PyPI..."
                    PIP_CONFIG_FILE=/dev/null pip install --no-deps gemmi meeko \
                        || warn "Meeko install failed - install it manually into the venv."
                fi
            fi
            # Restore numpy aliases (np.bool, ...) that the DD code needs but
            # modern numpy removed. Runs at every python startup in this env.
            _install_numpy_shim
            python -c "$REQUIRED_IMPORTS" 2>/dev/null \
                && success "Virtualenv ready and verified: $VENV_DIR" \
                || warn "Some imports still fail. Check the log above."
            deactivate 2>/dev/null || true
        else
            error "Failed to create virtualenv at $VENV_DIR."
        fi
    fi
fi

# Always leave the config with a working activation command.
[ -z "$ENV_ACTIVATE" ] && ENV_ACTIVATE="conda activate ${CONDA_ENV:-dd-env}"

# =============================================================================
# Step 8: Deep Docking parameters  (defaults + paper guidance)
# =============================================================================
echo ""
info "Deep Docking parameters (Gentile et al., Nature Protocols 2022)."
echo "  Press Enter to accept the recommended default for each."

note "total_iterations: active-learning rounds; more rounds shrink the library further."
prompt_default "total_iterations" "11"; TOTAL_ITER="$REPLY_VAL"

note "train_size: molecules sampled + docked for training each iteration."
prompt_default "train_size" "1000000"; TRAIN_SIZE="$REPLY_VAL"

note "val_size: validation and test set size, sampled once in iter 1. >=250,000 recommended."
prompt_default "val_size" "1000000"; VAL_SIZE="$REPLY_VAL"

note "percent_first / percent_last: top-scoring %% labelled 'virtual hit' in the first vs"
note "last iteration (tightened over the run so late models focus on the best binders)."
prompt_default "percent_first" "1.0"; PCT_FIRST="$REPLY_VAL"
prompt_default "percent_last" "0.01"; PCT_LAST="$REPLY_VAL"

note "recall: fraction of true virtual hits the model must keep when it sets its threshold"
note "(higher = fewer missed actives but a larger surviving set). 0.75-0.95 typical."
prompt_default "recall" "0.90"; RECALL="$REPLY_VAL"

note "num_models: hyperparameter models trained per iteration (grid search picks the best)."
prompt_default "num_models (16/24/48/72/144)" "24"; NUM_MODELS="$REPLY_VAL"

# =============================================================================
# Step 9: Write campaign.yaml
# =============================================================================
mkdir -p "$CAMPAIGN_DIR_EXPANDED"
CONFIG_FILE="$CAMPAIGN_DIR_EXPANDED/campaign.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if ! yes_no "campaign.yaml already exists. Overwrite?" "N"; then
        CONFIG_FILE="$CAMPAIGN_DIR_EXPANDED/campaign_$(date +%Y%m%d_%H%M%S).yaml"
    fi
fi

# Finalize env.modules: the docking chain from Step 6, plus (for a venv) the
# python module and, if one with rdDetermineBonds was found, the rdkit module,
# which must load before the venv activates. If no rdkit module qualified,
# RDKit came from PyPI and is not listed here. Order preserved: insert the extra
# modules just before the engine module (the last one).
FINAL_MODS=("${MODULE_ARR[@]}")
if [[ "$ENV_ACTIVATE" == *bin/activate* ]]; then
    EXTRA_MODS=()
    [ -n "${PY_MODULE:-}" ]    && EXTRA_MODS+=("$PY_MODULE")
    [ -n "${RDKIT_MODULE:-}" ] && EXTRA_MODS+=("$RDKIT_MODULE")
    if [ "${#EXTRA_MODS[@]}" -gt 0 ]; then
        n=${#MODULE_ARR[@]}
        if [ "$n" -gt 0 ]; then
            FINAL_MODS=("${MODULE_ARR[@]:0:n-1}" "${EXTRA_MODS[@]}" "${MODULE_ARR[@]:n-1}")
        else
            FINAL_MODS=("${EXTRA_MODS[@]}")
        fi
    fi
fi
MODULES_YAML="[]"
if [ "${#FINAL_MODS[@]}" -gt 0 ]; then
    MODULES_YAML=$(printf '"%s", ' "${FINAL_MODS[@]}"); MODULES_YAML="[${MODULES_YAML%, }]"
fi

# Engine-specific docking lines.
if [ "$DOCK_PROGRAM" = "GNINA" ]; then
    ENGINE_YAML=$(printf '  gnina_cnn: "%s"\n  exhaustiveness: %s' "$GNINA_CNN" "$EXHAUST")
else
    ENGINE_YAML=$(printf '  maps_fld: "%s"\n  autodock_bin: "%s"\n  autodock_nrun: %s' "$MAPS_FLD" "$AUTODOCK_BIN" "$AUTODOCK_NRUN")
fi

cat > "$CONFIG_FILE" << YAML
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
  receptor_file: "$RECEPTOR_FILE"
  box_json: "$BOX_JSON"                 # produced by dd_receptor_prep.py
  score_keyword: "$SCORE_KEYWORD"
  site:
$SITE_YAML
$ENGINE_YAML

dd:
  total_iterations: $TOTAL_ITER
  train_size: $TRAIN_SIZE
  val_size: $VAL_SIZE
  percent_first: $PCT_FIRST
  percent_last: $PCT_LAST
  recall: $RECALL
  num_models: $NUM_MODELS
  num_cpus_sampling: 60

scheduler:
  type: "SLURM"
  account: "$SLURM_ACCOUNT"
  cpu_partition: ""
  gpu_partition: ""

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
    phase3_docking:   {nodes: 1, cpus: 16, mem: "64G", gpus: 1}
    phase4_training:  {nodes: 1, cpus: 8,  mem: "32G", gpus: 1}
    phase5_inference: {nodes: 1, cpus: 8,  mem: "32G", gpus: 1}
    final_extraction: {nodes: 1, cpus: 60, mem: "32G", gpus: 0}

env:
  conda_env: "$CONDA_ENV"
  dd_protocol_dir: "$DD_PROTOCOL_DIR"
  modules: $MODULES_YAML
YAML

# Append the activation command as a single-quoted YAML scalar so its $(...) and
# quotes are preserved verbatim (expanded only when a job runs it).
printf "  activate: '%s'\n" "$ENV_ACTIVATE" >> "$CONFIG_FILE"

success "Config written to: $CONFIG_FILE"

# =============================================================================
# Step 10: Build the binding box / maps, then validate
# =============================================================================
if yes_no "Run dd_receptor_prep.py now to build the binding box${ADD:+ and maps}?" "Y"; then
    info "Preparing receptor (Gnina: box only; AutoDock-GPU: also grid maps, ~minutes)..."
    python3 "$SCRIPT_DIR/dd_receptor_prep.py" --config "$CONFIG_FILE" || \
        warn "Receptor prep did not finish; run it manually before launching."
fi

echo ""
info "Validating configuration..."
python3 "$SCRIPT_DIR/dd_validate.py" --config "$CONFIG_FILE" 2>&1 || true

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}   Setup complete${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Account       : $SLURM_ACCOUNT"
echo "  Engine        : $DOCK_PROGRAM"
echo "  Site strategy : $SITE_METHOD"
echo "  Modules       : $MODULES_YAML"
echo "  Conda env     : $CONDA_ENV"
echo "  DD protocol   : $DD_PROTOCOL_DIR"
echo "  Config        : $CONFIG_FILE"
echo ""
if [ "$SMILES_DIR" = "\$SCRATCH/library_prepared" ]; then
    echo "  Link a prepared library before launching:"
    echo -e "    ${BOLD}python dd_active_learning/dd_link.py --prep-work-dir <work_dir> --campaign $CONFIG_FILE${RESET}"
    echo ""
fi
echo "  Preview, then launch:"
echo -e "    ${BOLD}python dd_active_learning/dd_orchestrator.py --config $CONFIG_FILE --dry-run${RESET}"
echo -e "    ${BOLD}python dd_active_learning/dd_orchestrator.py --config $CONFIG_FILE${RESET}"
echo ""
echo "  Track progress:"
echo -e "    ${BOLD}python dd_active_learning/dd_status.py --config $CONFIG_FILE${RESET}"
echo ""
