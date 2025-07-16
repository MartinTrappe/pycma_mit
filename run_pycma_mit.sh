#!/usr/bin/env bash
# run_pycma_mit.sh
#
# Wrapper to create a virtual environment and execute pycma_mit.py.
#
# Author: Martin-Isbjörn Trappe
# Email: martin.trappe@quantumlah.org
# Date: 2025-07-04
# License: MIT License
#
# Usage:
#   ./run_pycma_mit.sh
#
# Requirements:
#   - bash
#   - python3 (≥3.7) with the `venv` module
#   - pip
#
# What it does:
#   1. Creates/activates venv in ./venv
#   2. Installs dependencies (cma, numpy, matplotlib, pandas)
#   3. Creates data folder relative to script path
#   4. Runs pycma_mit.py
#   5. Exits/deactivates the venv


# === Configuration ===
SCRIPT_NAME="pycma_mit.py"
VENV_DIR="$HOME/venvs/cma-env"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# === Create /data folder if not present ===
mkdir -p "$DATA_DIR"

# === Create virtual environment if needed ===
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# === Activate venv and install packages ===
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install numpy matplotlib pandas cma

# === Run the Python script ===
python "$SCRIPT_DIR/$SCRIPT_NAME"

# === Deactivate venv ===
deactivate

