#!/usr/bin/env bash
set -euo pipefail

# Simple helper to create a venv, install requirements, and run the Streamlit app.
# Usage: ./run.sh

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

if [ ! -x "$(command -v $PYTHON)" ]; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  $PYTHON -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Ensure pip is up-to-date
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run streamlit_app.py
