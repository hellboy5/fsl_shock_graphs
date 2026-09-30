#!/bin/bash
#SBATCH -J REPRODUCE_UNIMODAL_10K
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -o logs/reproduce_unimodal_10k_%j.out
#SBATCH -e logs/reproduce_unimodal_10k_%j.err

mkdir -p logs

source /users/mnarayan/shock-graph-reader/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo "Unimodal 10k Baseline Suite started on $(hostname) at $(date)"
echo "GPU assigned: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "=========================================================="

format_time() {
    local T=$1
    local H=$((T / 3600))
    local M=$(((T % 3600) / 60))
    local S=$((T % 60))
    printf "%02d:%02d:%02d" $H $M $S
}

run_unimodal() {
    local exp_name=$1
    local overrides=$2
    local save_dir="experiments/unimodal_baselines/${exp_name}"

    echo ""
    echo "=========================================================="
    echo ">>> Running: ${exp_name}"
    echo "    Save Dir: ${save_dir}"
    echo "=========================================================="

    mkdir -p "${save_dir}"

    # 1. Train (100 Epochs, 15-Way 15-Query)
    start_train=$(date +%s)
    python -u main.py \
        training.save_dir="${save_dir}" \
        ${overrides}
    end_train=$(date +%s)
    echo "--> Training finished in $(format_time $((end_train - start_train)))"

    ckpt="${save_dir}/checkpoints/best_model.pth"

    # 2. Eval: 5-Way 1-Shot (10,000 Episodes)
    echo "--> Running 5-Way 1-Shot Evaluation (10,000 episodes)..."
    start_eval1=$(date +%s)
    python -u main.py \
        mode=eval \
        task.n_way=5 \
        task.n_shot=1 \
        task.test_episodes=10000 \
        training.save_dir="${save_dir}" \
        checkpoint_path="${ckpt}" \
        ${overrides} \
        > "${save_dir}/eval_1shot_10k.txt" 2>&1
    end_eval1=$(date +%s)
    echo "--> 1-Shot Eval finished in $(format_time $((end_eval1 - start_eval1)))"

    # 3. Eval: 5-Way 5-Shot (10,000 Episodes)
    echo "--> Running 5-Way 5-Shot Evaluation (10,000 episodes)..."
    start_eval5=$(date +%s)
    python -u main.py \
        mode=eval \
        task.n_way=5 \
        task.n_shot=5 \
        task.test_episodes=10000 \
        training.save_dir="${save_dir}" \
        checkpoint_path="${ckpt}" \
        ${overrides} \
        > "${save_dir}/eval_5shot_10k.txt" 2>&1
    end_eval5=$(date +%s)
    echo "--> 5-Shot Eval finished in $(format_time $((end_eval5 - start_eval5)))"

    echo "Finished: ${exp_name}"
    grep -E "Final Test Results:" "${save_dir}/eval_1shot_10k.txt" | sed 's/^/  [1-Shot 10k] /' || true
    grep -E "Final Test Results:" "${save_dir}/eval_5shot_10k.txt" | sed 's/^/  [5-Shot 10k] /' || true
}

# 1. Vision Baseline (~59.6% 1-Shot, ~76.5% 5-Shot)
run_unimodal "Vision_ResNet12_Baseline" "model.modality=vision"

# 2. Graph Baseline (~42.5% 1-Shot, ~58.0% 5-Shot)
run_unimodal "Graph_GINE_640D_Baseline" "model.modality=graph model.graph_proj_dim=640"

# Summary Table
echo ""
echo "========================================================================================="
echo "                         UNIMODAL BASELINES (10,000 EPISODES) SUMMARY                    "
echo "========================================================================================="
printf "%-28s | %-24s | %-24s\n" "Experiment Name" "5-Way 1-Shot (±95% CI)" "5-Way 5-Shot (±95% CI)"
echo "-----------------------------------------------------------------------------------------"

for dir in experiments/unimodal_baselines/*; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        acc_1s=$(grep -oE "[0-9]+\.[0-9]+% ± [0-9]+\.[0-9]+%" "$dir/eval_1shot_10k.txt" | tail -n 1)
        acc_5s=$(grep -oE "[0-9]+\.[0-9]+% ± [0-9]+\.[0-9]+%" "$dir/eval_5shot_10k.txt" | tail -n 1)
        printf "%-28s | %-24s | %-24s\n" \
            "$name" "${acc_1s:-Failed}" "${acc_5s:-Failed}"
    fi
done
echo "========================================================================================="
echo "Completed at: $(date)"
echo "========================================================================================="

deactivate
