# CSAT Experiments - Server Execution Summary

## ✅ What's Ready

All files needed to run experiments 2-6 on your A100 server (wlmresgpu003) are ready:

### 1. SBATCH Scripts
- **[run_server_experiments.sh](file:///home/suraj/Git/CSAT/run_server_experiments.sh)** - Run experiments 2-6 sequentially
- **[run_single_server_experiment.sh](file:///home/suraj/Git/CSAT/run_single_server_experiment.sh)** - Run individual experiments

### 2. Experiment Configs  
- `configs/exp2_lr_tuning.yaml` - LR=0.00005
- `configs/exp3_loss_balanced.yaml` - Balanced loss weights
- `configs/exp4_architecture.yaml` - Queries=50, dropout=0.15  
- `configs/exp5_batch_training.yaml` - Batch=32
- `configs/exp6_optimized.yaml` - Combined optimized (50 epochs)

### 3. Code & Tools
- `train.py` - Modified with JSON logging
- `utils/experiment_logger.py` - Experiment tracking
- `compare_experiments.py` - Results analysis

## 🚀 Quick Start (3 Steps)

### Step 1: Transfer to Server

```bash
# From your local machine
rsync -avz --exclude 'runs/' --exclude 'logs/' --exclude '__pycache__/' \
  /home/suraj/Git/CSAT/ \
  skumar@wlmresgpu003:~/CSAT/
```

### Step 2: Submit Job

```bash
# SSH to server
ssh skumar@wlmresgpu003

# Navigate to project
cd ~/CSAT

# Submit experiments 2-6
sbatch run_server_experiments.sh
```

### Step 3: Monitor

```bash
# Check job status
squeue -u $USER

# Watch output (replace JOB_ID with actual ID from squeue)
tail -f logs/slurm_exp2-6_<JOB_ID>.out

# Or watch specific experiment
tail -f logs/exp2_lr_tuning_output.txt
```

## ⏱️ Expected Timeline

On A100 GPU:
- Experiments 2-5: ~10-20 min each (20 epochs)
- Experiment 6: ~25-50 min (50 epochs)
- **Total: ~1.5 - 2.5 hours**

## 📊 After Completion

### Download Results

```bash
# From local machine
rsync -avz skumar@wlmresgpu003:~/CSAT/logs/ /home/suraj/Git/CSAT/logs/
rsync -avz skumar@wlmresgpu003:~/CSAT/runs/ /home/suraj/Git/CSAT/runs/
```

### Compare Experiments

```bash
# Install tabulate if needed
pip install tabulate

# Analyze results
python compare_experiments.py
```

This will show:
- Best performing configuration
- Which experiments enabled learning
- Recommended hyperparameters

## 📧 Notifications

You'll receive emails at sk1019@nemours.org when:
- Job starts (BEGIN)
- Job completes (END)
- Job fails (FAIL)

## 📁 Output Files

After job completes:
```
logs/
├── exp2_lr_tuning_*.json        # Metrics per epoch
├── exp2_lr_tuning_output.txt    # Full training log
├── exp3_loss_balanced_*.json
├── exp3_loss_balanced_output.txt
├── ... (for experiments 4, 5, 6)
└── slurm_exp2-6_<JOB_ID>.out   # SLURM job output

runs/
├── exp2_lr_tuning/detection_best.pth
├── exp3_loss_balanced/detection_best.pth
└── ... (for experiments 4, 5, 6)
```

## 🔧 Troubleshooting

See [SERVER_GUIDE.md](file:///home/suraj/Git/CSAT/SERVER_GUIDE.md) for:
- Environment setup
- Job monitoring commands
- Error handling
- Individual experiment submission

## 📖 Full Documentation

- **[SERVER_GUIDE.md](file:///home/suraj/Git/CSAT/SERVER_GUIDE.md)** - Complete server usage guide
- **[EXPERIMENTS_GUIDE.md](file:///home/suraj/Git/CSAT/EXPERIMENTS_GUIDE.md)** - General experiments guide
- **[walkthrough.md](file:///home/suraj/.gemini/antigravity/brain/00204518-294b-4682-9a7c-4c1e1e4a430a/walkthrough.md)** - Full setup walkthrough

---

**Next**: Transfer files and submit the SBATCH job! 🎯
