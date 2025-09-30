#!/bin/bash

# Set absolute paths for the script
readonly SCRIPT_DIR="/home/Competition2025/P05/P05U001/team_suzuki/data/data_generation/sft_data_generation/scripts"
readonly PROJECT_DIR="/home/Competition2025/P05/P05U001/team_suzuki/data/data_generation/sft_data_generation"

echo "Setting up environment for SFT data generation..."

# Environment path
CONDA_ENV_PATH="/home/Competition2025/P05/P05U001/envs/datagen"
# Check if environment.yaml exists
if [ ! -f "$PROJECT_DIR/environment.yaml" ]; then
    echo "Error: environment.yaml not found in $PROJECT_DIR"
    exit 1
fi

# Create or update conda environment from environment.yaml
echo "Creating/updating conda environment at $CONDA_ENV_PATH..."
if conda env list | grep -q "^${CONDA_ENV_PATH}"; then
    echo "[INFO] Environment exists -> updating"
    conda env update -p "$CONDA_ENV_PATH" -f "$PROJECT_DIR/environment.yaml" --prune
else
    echo "[INFO] Environment not found -> creating"
    conda env create -p "$CONDA_ENV_PATH" -f "$PROJECT_DIR/environment.yaml"
fi

# Activate environment
source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PATH"

# Create necessary directories
echo "Creating directories..."
mkdir -p /home/Competition2025/P05/P05U001/datasets/sft_dataset
mkdir -p /home/Competition2025/P05/P05U001/logs

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $CONDA_ENV_PATH"
echo ""
echo "To submit a job, run:"
echo "  sbatch ${SCRIPT_DIR}/run_datagen_job.sh"