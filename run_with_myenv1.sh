#!/bin/bash
# Helper script to run Python scripts with myenv1 conda environment
# Usage: ./run_with_myenv1.sh your_script.py [args...]

CONDA_ENV_PATH="/Users/zakaria/anaconda3/envs/myenv1"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Conda environment myenv1 not found at $CONDA_ENV_PATH"
    echo "Please check your conda environment path."
    exit 1
fi

# Run the script with the conda environment's Python
"$PYTHON_BIN" "$@"

