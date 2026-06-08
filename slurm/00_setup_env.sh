#!/bin/bash
# =============================================================================
# 00_setup_env.sh — One-time environment setup on DRA cluster
#
# Run this ONCE interactively on the login node BEFORE submitting any jobs.
# It creates a Python virtual environment in $SCRATCH and installs dd_prep.
#
# Usage:
#   bash slurm/00_setup_env.sh
#
# After this script completes, all SLURM jobs will source this environment
# automatically.
# =============================================================================

set -euo pipefail

# ── 1. Customise these two variables ─────────────────────────────────────────

# Path to your dd_prep project (the folder containing pyproject.toml).
PROJECT_DIR="$HOME/projects/dd_prep"

# Where the virtual environment will live.  $SCRATCH is the fast parallel
# filesystem on DRA — much better than $HOME for large installs.
VENV_DIR="$SCRATCH/dd_prep_venv"

# ── 2. Load required modules ──────────────────────────────────────────────────
# Module names are the same across all DRA clusters (Beluga, Cedar, Graham,
# Narval).  If a module is not found, check: module spider python
module purge
module load StdEnv/2023
module load python/3.11

echo "Python: $(which python) — $(python --version)"

# ── 3. Create the virtual environment ────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "Delete it and re-run if you want a clean install."
else
    python -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── 4. Install Python dependencies ───────────────────────────────────────────
pip install --upgrade pip setuptools wheel --quiet

# RDKit wheels are available on PyPI since RDKit 2022.
pip install pyyaml pandas rdkit tqdm --quiet

# Install dd_prep itself in editable mode so code changes take effect
# without reinstalling.
pip install -e "$PROJECT_DIR" --quiet

echo ""
echo "Installation complete."
python -c "import dd_prep; print('dd_prep version:', dd_prep.__version__)"

# ── 5. OpenEye licence check ──────────────────────────────────────────────────
# OpenEye tools (flipper, tautomers, oeomega) must be installed separately
# and an OE_LICENSE file must be available on all compute nodes.
#
# The licence file is typically stored in your $HOME (accessible everywhere):
#   export OE_LICENSE=$HOME/oe_license.txt
#
# Test that the licence is valid:
if [ -n "${OE_LICENSE:-}" ] && [ -f "$OE_LICENSE" ]; then
    echo "OE_LICENSE found: $OE_LICENSE"
    if command -v flipper &>/dev/null; then
        echo "flipper found: $(which flipper)"
    else
        echo "WARNING: flipper not found on PATH."
        echo "Add your OpenEye bin directory to PATH in your ~/.bashrc:"
        echo "  export PATH=\$HOME/openeye/bin:\$PATH"
    fi
else
    echo ""
    echo "WARNING: OE_LICENSE is not set or the file does not exist."
    echo "Add this to your ~/.bashrc:"
    echo "  export OE_LICENSE=\$HOME/oe_license.txt"
    echo ""
    echo "The filter, split, organize, and fingerprint steps do NOT require"
    echo "OpenEye.  Only flipper, tautomers, and omega need a licence."
fi

echo ""
echo "Setup complete.  You can now submit jobs with:"
echo "  bash slurm/submit_pipeline.sh --config my_config.yaml"