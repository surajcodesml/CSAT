#!/bin/bash
# Fully automated experiment runner - no user interaction required
# Run with: nohup ./run_all_experiments.sh > logs/full_run.log 2>&1 &

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate oct

echo "=========================================="
echo "CSAT Hyperparameter Optimization"
echo "Automated Sequential Execution"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Create output directories
mkdir -p logs runs

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local config_file="configs/${exp_name}.yaml"
    
    echo ""
    echo "=========================================="
    echo "Starting: $exp_name"
    echo "Time: $(date)"
    echo "Description: $exp_desc"
    echo "Config: $config_file"
    echo "=========================================="
    echo ""
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file $config_file not found!"
        return 1
    fi
    
    # Run training and save output
    python train.py --config "$config_file" 2>&1 | tee "logs/${exp_name}_output.txt"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ COMPLETED: $exp_name ($(date))"
        echo ""
    else
        echo ""
        echo "✗ FAILED: $exp_name with exit code $exit_code ($(date))"
        echo ""
        return $exit_code
    fi
}

# Record start time
START_TIME=$(date +%s)

# Run all 6 experiments sequentially
run_experiment "exp1_baseline" "Baseline with current hyperparameters (20 epochs)"
run_experiment "exp2_lr_tuning" "Reduced learning rate 0.00005 (20 epochs)"
run_experiment "exp3_loss_balanced" "Balanced loss weights (20 epochs)"
run_experiment "exp4_architecture" "Reduced queries=50, dropout=0.15 (20 epochs)"
run_experiment "exp5_batch_training" "Reduced batch size to 32 (20 epochs)"
run_experiment "exp6_optimized" "Combined optimized config (50 epochs)"

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Finished: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved in:"
echo "  - logs/*.txt (training outputs)"
echo "  - logs/*.json (experiment metrics)"
echo "  - runs/exp*/ (model checkpoints)"
echo ""
echo "To compare results:"
echo "  python compare_experiments.py"
echo "=========================================="
