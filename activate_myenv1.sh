#!/bin/bash
# Quick activation script - source this to activate myenv1
# Usage: source activate_myenv1.sh

export PATH="/Users/zakaria/anaconda3/envs/myenv1/bin:$PATH"
export VIRTUAL_ENV="/Users/zakaria/anaconda3/envs/myenv1"
export CONDA_DEFAULT_ENV="myenv1"

echo "✓ myenv1 environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
