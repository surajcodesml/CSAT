#!/bin/bash
# Run remaining experiments (2-6) after experiment 1 completes
# Use this if experiment 1 is already running separately

set -e

eval "$(conda shell.bash hook)"
conda activate oct

echo "=========================================="
echo "Running Experiments 2-6"
echo "Started: $(date)"
echo "=========================================="
echo ""

mkdir -p logs runs

run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local config_file="configs/${exp_name}.yaml"
    
    echo ""
    echo "=========================================="
    echo "Starting: $exp_name ($(date))"
    echo "Description: $exp_desc"
    echo "=========================================="
    echo ""
    
    python train.py --config "$config_file" 2>&1 | tee "logs/${exp_name}_output.txt"
    
    echo "✓ Completed: $exp_name ($(date))"
}

START_TIME=$(date +%s)

run_experiment "exp2_lr_tuning" "Reduced LR=0.00005"
run_experiment "exp3_loss_balanced" "Balanced loss weights"  
run_experiment "exp4_architecture" "Queries=50, dropout=0.15"
run_experiment "exp5_batch_training" "Batch size=32"
run_experiment "exp6_optimized" "Combined optimized (50 epochs)"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "Experiments 2-6 completed in $((TOTAL_TIME / 3600))h $(((TOTAL_TIME % 3600) / 60))m"
echo "Run: python compare_experiments.py"
