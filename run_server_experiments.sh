#!/bin/bash
#SBATCH --job-name="CSAT_Exp2-6"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=300g
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sk1019@nemours.org
#SBATCH --output=logs/slurm_exp2-6_%j.out
#SBATCH --error=logs/slurm_exp2-6_%j.err

export OMP_NUM_THREADS=1
hn=$(hostname -s)

if [[ $hn == "wlmresgpu003" ]]; then
    echo "Running on $hn with vision env"
    eval "$(conda shell.bash hook)"
    source /home/skumar/miniconda3/etc/profile.d/conda.sh 
    conda activate vision
else
    echo "Must be run on gpu3!"
    exit 1
fi

echo "=========================================="
echo "CSAT Hyperparameter Experiments 2-6"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Create log directory
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

# Run experiments 2-6 sequentially
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
echo "EXPERIMENTS 2-6 COMPLETED!"
echo "=========================================="
echo "Finished: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Results saved in:"
echo "  - logs/*.txt (training outputs)"
echo "  - logs/*.json (experiment metrics)"
echo "  - runs/exp*/ (model checkpoints)"
echo ""
echo "To compare results:"
echo "  python compare_experiments.py"
echo "=========================================="

exit
