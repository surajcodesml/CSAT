#!/bin/bash
# Automated experiment runner for CSAT hyperparameter optimization
# Runs 6 experiments sequentially and logs results

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate oct

# Create output directories
mkdir -p logs runs

# Define experiments
declare -a experiments=(
    "exp1_baseline"
    "exp2_lr_tuning"
    "exp3_loss_balanced"
    "exp4_architecture"
    "exp5_batch_training"
    "exp6_optimized"
)

declare -a descriptions=(
    "Baseline with current hyperparameters"
    "Reduced learning rate (0.00005)"
    "Balanced loss weights"
    "Reduced num_queries=50, dropout=0.15"
    "Reduced batch size to 32"
    "Combined optimized configuration"
)

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local config_file="configs/${exp_name}.yaml"
    
    echo ""
    echo "=========================================="
    echo "Starting: $exp_name"
    echo "Description: $exp_desc"
    echo "Config: $config_file"
    echo "=========================================="
    echo ""
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file $config_file not found!"
        return 1
    fi
    
    # Run training and save output to log file
    python train.py --config "$config_file" 2>&1 | tee "logs/${exp_name}_output.txt"
    
    echo ""
    echo "Completed: $exp_name"
    echo ""
}

# Main execution
echo "========================================"
echo "CSAT Hyperparameter Optimization"
echo "Running 6 experiments sequentially"
echo "========================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run experiments 1-5
for i in {0..4}; do
    run_experiment "${experiments[$i]}" "${descriptions[$i]}"
    
    # Ask user if they want to continue after each experiment
    echo ""
    read -p "Continue to next experiment? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping experiments at user request."
        exit 0
    fi
done

# Prompt before running experiment 6 (longer run)
echo ""
echo "=========================================="
echo "EXPERIMENT 6: Combined Optimized Config"
echo "This will run for 50 epochs (longer)"
echo "=========================================="
echo ""
read -p "Run experiment 6? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_experiment "${experiments[5]}" "${descriptions[5]}"
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved in logs/ directory"
echo "JSON logs saved in logs/ directory"
echo "Model checkpoints in runs/exp*/ directories"
echo "========================================"
echo ""
echo "To compare results, run:"
echo "  python compare_experiments.py"
