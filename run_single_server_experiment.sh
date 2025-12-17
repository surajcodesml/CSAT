#!/bin/bash
#SBATCH --job-name="CSAT_SingleExp"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=300g
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sk1019@nemours.org
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Single experiment runner - specify experiment number as argument
# Usage: sbatch --export=EXP_NUM=2 run_single_server_experiment.sh

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

# Default to experiment 2 if not specified
EXP_NUM=${EXP_NUM:-2}

echo "=========================================="
echo "CSAT Experiment ${EXP_NUM}"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

mkdir -p logs runs

# Determine which experiment to run
case $EXP_NUM in
    2)
        CONFIG="configs/exp2_lr_tuning.yaml"
        DESC="Learning Rate Tuning (LR=0.00005)"
        ;;
    3)
        CONFIG="configs/exp3_loss_balanced.yaml"
        DESC="Balanced Loss Weights"
        ;;
    4)
        CONFIG="configs/exp4_architecture.yaml"
        DESC="Architecture Tuning (queries=50, dropout=0.15)"
        ;;
    5)
        CONFIG="configs/exp5_batch_training.yaml"
        DESC="Batch Size Reduction (batch=32)"
        ;;
    6)
        CONFIG="configs/exp6_optimized.yaml"
        DESC="Combined Optimized Config (50 epochs)"
        ;;
    *)
        echo "ERROR: Invalid experiment number: $EXP_NUM"
        echo "Valid options: 2, 3, 4, 5, 6"
        exit 1
        ;;
esac

echo "Config: $CONFIG"
echo "Description: $DESC"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file $CONFIG not found!"
    exit 1
fi

# Run training
python train.py --config "$CONFIG" 2>&1 | tee "logs/exp${EXP_NUM}_output.txt"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment ${EXP_NUM} COMPLETED"
else
    echo "✗ Experiment ${EXP_NUM} FAILED (exit code: $EXIT_CODE)"
fi
echo "Finished: $(date)"
echo "=========================================="

exit $EXIT_CODE
