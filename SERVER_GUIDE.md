# Running CSAT Experiments on SLURM Server

## Quick Start

### Option 1: Run All Experiments 2-6 Sequentially (Recommended)

```bash
# Submit the job
sbatch run_server_experiments.sh

# Check status
squeue -u $USER

# Monitor output
tail -f logs/slurm_exp2-6_<JOB_ID>.out

# Or monitor specific experiment
tail -f logs/exp2_lr_tuning_output.txt
```

### Option 2: Run Individual Experiments

Submit experiments individually (useful if you want to run them in parallel on different GPUs or test one first):

```bash
# Run experiment 2
sbatch --export=EXP_NUM=2 run_single_server_experiment.sh

# Run experiment 3
sbatch --export=EXP_NUM=3 run_single_server_experiment.sh

# Run experiment 4
sbatch --export=EXP_NUM=4 run_single_server_experiment.sh

# Run experiment 5
sbatch --export=EXP_NUM=5 run_single_server_experiment.sh

# Run experiment 6 (longer - 50 epochs)
sbatch --export=EXP_NUM=6 run_single_server_experiment.sh
```

## Before Running on Server

### 1. Transfer Files to Server

From your local machine, sync the repository to the server:

```bash
# From local machine
rsync -avz --exclude 'runs/' --exclude 'logs/' --exclude '__pycache__/' \
  /home/suraj/Git/CSAT/ \
  skumar@wlmresgpu003:/path/to/CSAT/
```

**Important files to transfer:**
- `configs/` directory (all 6 experiment configs)
- `utils/experiment_logger.py` (JSON logging)
- `train.py` (modified with logger integration)
- `run_server_experiments.sh` (SBATCH script)
- `run_single_server_experiment.sh` (optional)
- `compare_experiments.py` (for results analysis)
- All other code files

### 2. Verify Environment on Server

SSH to server and check:

```bash
ssh skumar@wlmresgpu003

# Check conda environment
conda activate vision

# Verify required packages
python -c "import torch; print(torch.__version__)"
python -c "import yaml; print('YAML OK')"

# Check if CSAT code is there
cd /path/to/CSAT
ls configs/  # Should show exp*.yaml files
```

### 3. Ensure Data is Available

Make sure the data files are accessible on the server:

```bash
# Check data files
ls pickle/  # Should have .pkl files
ls data/train.txt data/val.txt  # Should have file lists
ls model/best_pretrainer.pth  # Should have pretrained weights
```

## Monitoring Jobs

### Check Job Status
```bash
# View your jobs
squeue -u $USER

# View job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>
```

### Monitor Progress

```bash
# Watch SLURM output
tail -f logs/slurm_exp2-6_*.out

# Watch specific experiment
tail -f logs/exp2_lr_tuning_output.txt

# Check JSON metrics (per epoch)
cat logs/exp2_lr_tuning_*.json | tail -50

# Monitor GPU usage (if you have access)
watch -n 2 nvidia-smi
```

### Check Completion

```bash
# List completed checkpoints
ls -lh runs/exp*/detection_best.pth

# List JSON logs
ls -lh logs/exp*.json

# Check which experiments finished
ls logs/*_output.txt
```

## Expected Timeline on A100 GPU

With A100 GPU (faster than local):
- **Experiments 2-5**: 20 epochs × ~30-60 sec/epoch = ~10-20 min each
- **Experiment 6**: 50 epochs × ~30-60 sec/epoch = ~25-50 min
- **Total**: ~1.5 - 2.5 hours for experiments 2-6

## After Experiments Complete

### Download Results to Local

```bash
# From local machine
rsync -avz skumar@wlmresgpu003:/path/to/CSAT/logs/ \
  /home/suraj/Git/CSAT/logs/

rsync -avz skumar@wlmresgpu003:/path/to/CSAT/runs/ \
  /home/suraj/Git/CSAT/runs/
```

### Compare Results

Either on server or locally:

```bash
# Install tabulate if needed (on server)
pip install tabulate

# Compare experiments
python compare_experiments.py
```

## Troubleshooting

### Job Not Starting
```bash
# Check job queue
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# Check partition availability
sinfo -p gpu3
```

### Out of Memory
The configs are set for reasonable batch sizes, but if OOM occurs:
- Experiment 5 already uses batch=32 (smallest)
- Can reduce batch further by editing config

### Environment Issues
```bash
# If conda env missing packages
conda activate vision
pip install pyyaml tabulate

# If vision env doesn't exist, create it
conda create -n vision python=3.9
conda activate vision
pip install torch torchvision torchaudio pyyaml tabulate tqdm scipy scikit-learn
```

### Check Logs
```bash
# SLURM output
cat logs/slurm_exp2-6_<JOB_ID>.out

# SLURM errors
cat logs/slurm_exp2-6_<JOB_ID>.err

# Training output
cat logs/exp2_lr_tuning_output.txt
```

## Files Created for Server

- `run_server_experiments.sh` - Main SBATCH script for experiments 2-6
- `run_single_server_experiment.sh` - Individual experiment runner
- This guide - `SERVER_GUIDE.md`

## Notes

- The SBATCH scripts use your existing server configuration (wlmresgpu003, vision env)
- All experiments will run with `world_size=1` (single GPU) as configured
- JSON logs will be created just like local runs
- Email notifications will be sent (BEGIN, END, FAIL) to sk1019@nemours.org
