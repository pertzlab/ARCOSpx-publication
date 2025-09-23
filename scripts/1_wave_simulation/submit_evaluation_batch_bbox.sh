#!/bin/bash
#SBATCH --job-name=eval_%a
#SBATCH --output=logs/out/eval_%a.out
#SBATCH --error=logs/err/eval_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --array=0-239%24     # 240 combos, throttle to 24 at once
#SBATCH --nodes=1

# ----------------- dimensions -----------------
num_sim_functions=4
snr_vals=(-15 -10 -5 0 25 inf)
iterations=10
snr_count=${#snr_vals[@]}
combos_per_sim=$(( snr_count * iterations ))

task=${SLURM_ARRAY_TASK_ID}

sim_function_index=$(( task / combos_per_sim ))
tmp=$(( task % combos_per_sim ))
snr_index=$(( tmp / iterations ))
iteration=$(( tmp % iterations ))

snr_val=${snr_vals[$snr_index]}

echo "TASK=$task  sim=$sim_function_index  snr=$snr_val  iter=$iteration"

# ---- avoid BLAS oversubscription (optional) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --------------- conda env ---------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate || true
conda activate pyimagej
echo "Env: $(conda info --envs | awk '/\\*/{print $1}')"

# -------------- run both trackers --------------
trackers=("arcospx" "trackmate")

for trk in "${trackers[@]}"; do
    echo "Evaluating tracker=$trk"
    python evaluation_metrics_bbox.py \
        --sim_function_index "$sim_function_index" \
        --snr "$snr_val" \
        --iteration "$iteration" \
        --tracker "$trk"
done

conda deactivate
