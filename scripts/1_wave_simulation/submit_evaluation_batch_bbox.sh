#!/bin/bash
#SBATCH --job-name=eval_run_%a
#SBATCH --output=eval_run_%a.out
#SBATCH --error=eval_run_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-23%12  # 4 simulation functions * 6 SNR values = 24 combinations, with max 6 running concurrently

# Define SNR values
snr_vals=(-15 -10 -5 0 25 "inf")  # Updated SNR values

# Calculate the simulation function index and SNR index from the job array index
sim_function_index=$(($SLURM_ARRAY_TASK_ID / ${#snr_vals[@]}))
snr_index=$(($SLURM_ARRAY_TASK_ID % ${#snr_vals[@]}))

# Get the SNR value for this job
snr_val=${snr_vals[$snr_index]}

source ~/miniforge3/etc/profile.d/conda.sh # this should point to the correct location of conda.sh in your system
conda deactivate # Deactivate any existing Conda environment
conda activate arcos_wave_eval
echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the evaluation and metrics computation for the specific combination
echo "Starting evaluation and metrics computation for SIM_FUNCTION_INDEX=${sim_function_index} with SNR=${snr_val}"
python evaluation_metrics_bbox.py --sim_function_index $sim_function_index --snr $snr_val

# Deactivate the Conda environment
conda deactivate
