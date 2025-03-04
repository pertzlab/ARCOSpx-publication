#!/bin/bash
#SBATCH --job-name=process_field_of_view
#SBATCH --array=0-111%10
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

source ~/miniforge3/etc/profile.d/conda.sh # Adjust the path as per your installation
# Activate the Conda environment
conda deactivate
conda activate imageanalysis # Adjust the environment name as per your setup
echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the Python script
python convpaint_batch.py
