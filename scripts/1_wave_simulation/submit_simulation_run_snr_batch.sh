#!/bin/bash
#SBATCH --job-name=sim_run_%a
#SBATCH --output=logs/out/sim_run_%a.out
#SBATCH --error=logs/err/sim_run_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=all
#SBATCH --array=0-239%12          # 4 sims * 6 snr * 10 iterations
#SBATCH --nodes=1

# ---------------- params -----------------
num_sim_functions=4
snr_vals=(-15 -10 -5 0 25 inf)
iterations=10
snr_count=${#snr_vals[@]}
combos_per_sim=$(( snr_count * iterations ))

# --------------- index math --------------
task=${SLURM_ARRAY_TASK_ID}

sim_function_index=$(( task / combos_per_sim ))
tmp=$(( task % combos_per_sim ))
snr_index=$(( tmp / iterations ))
iteration=$(( tmp % iterations ))

# -------------- validation ---------------
if (( sim_function_index >= num_sim_functions )); then
    echo "Invalid sim_function_index: $sim_function_index" >&2
    exit 1
fi
if (( snr_index >= snr_count )); then
    echo "Invalid snr_index: $snr_index" >&2
    exit 1
fi

snr_val=${snr_vals[$snr_index]}

echo "TASK: $task"
echo "sim_function_index: $sim_function_index"
echo "snr_index: $snr_index  -> $snr_val"
echo "iteration: $iteration"

# --------------- conda env ---------------
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate || true
conda activate pyimagej
echo "Env: $(conda info --envs | awk '/\*/{print $1}')"

# --------------- run ----------------------
echo "Running python simulation_tracking.py --sim_function_index $sim_function_index --snr $snr_val --iteration $iteration"
python simulation_tracking.py \
    --sim_function_index "$sim_function_index" \
    --snr "$snr_val" \
    --iteration "$iteration"

conda deactivate
