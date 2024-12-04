#!/bin/bash
#SBATCH --job-name=eval_run_%a
#SBATCH --output=eval_run_%a.out
#SBATCH --error=eval_run_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-23%12  # 4 simulation functions * 6 SNR values = 24 combinations, with max 6 running concurrently

# Define the number of simulation functions and SNR values
num_sim_functions=4
snr_vals=(-15 -10 -5 0 25 "inf")  # Updated SNR values

# Calculate the simulation function index and SNR index from the job array index
sim_function_index=$(($SLURM_ARRAY_TASK_ID / ${#snr_vals[@]}))
snr_index=$(($SLURM_ARRAY_TASK_ID % ${#snr_vals[@]}))

# Get the SNR value for this job
snr_val=${snr_vals[$snr_index]}

# Activate the Conda environment
source activate arcos_wave_eval

# Change to the directory where the evaluation scripts are located
cd ~/myimaging/arcos_wave_eval

# Run the evaluation and metrics computation for the specific combination
echo "Starting evaluation and metrics computation for SIM_FUNCTION_INDEX=${sim_function_index} with SNR=${snr_val}"
python evaluation_metrics_bbox.py $sim_function_index $snr_val

# Deactivate the Conda environment
conda deactivate
