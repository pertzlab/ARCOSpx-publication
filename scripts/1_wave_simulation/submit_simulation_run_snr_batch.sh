#!/bin/bash
#SBATCH --job-name=sim_run_%a
#SBATCH --output=sim_run_%a.out
#SBATCH --error=sim_run_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-23%12  # 4 simulation functions * 6 SNR values = 24 combinations

# Define SNR values
snr_vals=(-15 -10 -5 0 25 "inf")

# Calculate the simulation function index and SNR index from the job array index
sim_function_index=$(($SLURM_ARRAY_TASK_ID / ${#snr_vals[@]}))
snr_index=$(($SLURM_ARRAY_TASK_ID % ${#snr_vals[@]}))

# Validate indices
if [ $sim_function_index -ge $num_sim_functions ]; then
    echo "Invalid sim_function_index: $sim_function_index"
    exit 1
fi

if [ $snr_index -ge ${#snr_vals[@]} ]; then
    echo "Invalid snr_index: $snr_index"
    exit 1
fi

# Get the SNR value for this job
snr_val=${snr_vals[$snr_index]}

# Echo the parameters
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Simulation Function Index: $sim_function_index"
echo "SNR Index: $snr_index"
echo "SNR Value: $snr_val"

source ~/miniforge3/etc/profile.d/conda.sh # this should point to the correct location of conda.sh in your system
conda deactivate
conda activate imageanalysis
echo "Activated conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the simulation and tracking for the specific combination
echo "Starting simulation and tracking for SIM_FUNCTION_INDEX=${sim_function_index} with SNR=${snr_val}"
python simulation_tracking.py --sim_function_index $sim_function_index --snr $snr_val

# Deactivate the Conda environment
conda deactivate
