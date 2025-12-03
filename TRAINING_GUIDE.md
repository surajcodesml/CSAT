# CSAT Detector Training Guide

## Configuration File

All training parameters are now managed through `config.yaml`. This eliminates the need to modify code for different experiments.

## Quick Start

### Using Default Configuration
```bash
python train.py --config config.yaml
# or simply
python train.py
```

### Using the Quick Start Script
```bash
./start_training.sh
```

## Configuration Structure

The `config.yaml` file is organized into sections:

### 1. Paths
```yaml
paths:
  root: './'                              # Project root directory
  dataroot: './pickle'                    # Directory containing pickle files
  pretrain_weights: 'model/best_pretrainer.pth'  # Pretrained weights
  output_dir: 'runs/'                     # Output directory for checkpoints
  train_data: 'data/train.txt'           # Training data list
  val_data: 'data/val.txt'               # Validation data list
```

### 2. Model Architecture
```yaml
model:
  hidden_dim: 256                        # Transformer hidden dimension
  num_encoder_layers: 6                  # Number of encoder layers
  num_decoder_layers: 6                  # Number of decoder layers
  nheads: 8                              # Number of attention heads
  num_queries: 48                        # Number of object queries
  num_classes: 2                         # Number of classes (Fovea, SCR)
  dropout: 0.1                           # Dropout rate
```

### 3. Training Parameters
```yaml
training:
  epochs: 100                            # Number of training epochs
  train_batch_size: 64                   # Training batch size
  val_batch_size: 64                     # Validation batch size
  learning_rate: 0.0001                  # Initial learning rate
  weight_decay: 0.0001                   # L2 regularization
  
  lr_scheduler:
    type: 'ReduceLROnPlateau'            # Scheduler type
    patience: 2                          # Epochs without improvement
    cooldown: 2                          # Cooldown after LR reduction
    factor: 0.1                          # LR reduction factor
    mode: 'min'                          # Monitor min or max metric
```

### 4. Data Loading
```yaml
data:
  r: 3                                   # Number of adjacent frames to stack
  space: 1                               # Stride between adjacent frames
  num_workers: 6                         # DataLoader workers
  cache: 'ram'                           # Cache strategy ('ram', 'disk', false)
  shuffle: true                          # Shuffle training data
```

### 5. Distributed Training
```yaml
distributed:
  world_size: 2                          # Number of GPUs
  backend: 'nccl'                        # PyTorch distributed backend
  init_method: 'tcp://127.0.0.1:12426'  # Initialization method
  timeout: 5000                          # Timeout in seconds
```

### 6. Pretrained Weights
```yaml
pretrain:
  use_pretrained: true                   # Load pretrained encoder weights
  freeze_encoder: false                  # Freeze encoder during training
```

### 7. Resume Training
```yaml
resume:
  enabled: false                         # Resume from checkpoint
  weights_path: 'runs/outputs/detection_best.pth'  # Checkpoint path
```

### 8. Weights & Biases
```yaml
wandb:
  enabled: false                         # Enable W&B logging
  project: 'scr'                         # W&B project name
  name: 'train'                          # W&B run name
```

## Command Line Overrides

You can override specific config values from the command line:

```bash
# Override number of epochs
python train.py --epochs 50

# Override batch size
python train.py --batch_size 32

# Override learning rate
python train.py --lr 0.0005

# Override world size (number of GPUs)
python train.py --world_size 1

# Combine multiple overrides
python train.py --epochs 50 --batch_size 32 --lr 0.0005
```

## Common Use Cases

### 1. Training with Pretrained Weights (Default)
```bash
python train.py
```
This uses the pretrained encoder from `model/best_pretrainer.pth`.

### 2. Training from Scratch
Edit `config.yaml`:
```yaml
pretrain:
  use_pretrained: false
```
Then run:
```bash
python train.py
```

### 3. Fine-tuning with Frozen Encoder
Edit `config.yaml`:
```yaml
pretrain:
  use_pretrained: true
  freeze_encoder: true
```

### 4. Resume Training
Edit `config.yaml`:
```yaml
resume:
  enabled: true
  weights_path: 'runs/outputs/detection_best.pth'
```

### 5. Single GPU Training
```bash
python train.py --world_size 1
```

### 6. Quick Experiment with Different Learning Rate
```bash
python train.py --lr 0.001
```

### 7. Short Training Run for Testing
```bash
python train.py --epochs 5 --batch_size 16
```

## Output Structure

Training outputs are saved in the `runs/` directory:
```
runs/
└── outputs/
    ├── detection_best.pth    # Best model checkpoint
    └── detection_last.pth    # Last epoch checkpoint
```

Each checkpoint contains:
- `epoch`: Training epoch
- `model_state_dict`: Model weights
- `lr_state_dict`: Learning rate scheduler state
- `fitness`: Model fitness score
- `best_fitness`: Best fitness achieved

## Monitoring Training

### Console Output
Training progress is displayed in the console with:
- Loss values (CE loss, BB loss, IoU loss)
- GPU memory usage
- Validation metrics
- Learning rate updates

### Weights & Biases (Optional)
Enable W&B in `config.yaml`:
```yaml
wandb:
  enabled: true
  project: 'your_project_name'
  name: 'experiment_name'
```

## Troubleshooting

### Out of Memory
Reduce batch size in `config.yaml`:
```yaml
training:
  train_batch_size: 32
  val_batch_size: 32
```

### Slow Data Loading
Adjust num_workers:
```yaml
data:
  num_workers: 4  # Reduce if you have fewer CPU cores
```

### Training Not Improving
Try adjusting the learning rate:
```yaml
training:
  learning_rate: 0.00005  # Lower learning rate
```

Or modify the scheduler settings:
```yaml
training:
  lr_scheduler:
    patience: 5  # Wait longer before reducing LR
```

## Data Preparation

Before training, ensure you have:
1. Pickle files in `./pickle/` directory (12,623 files)
2. Generated `data/train.txt` and `data/val.txt` (already done)

If you need to regenerate the splits:
```bash
python -c "from utils.util import get_pickles, ids, dataset_trainer; \
import random; random.seed(1027); \
pkl_files = get_pickles('pickle'); \
f, p, t = ids(pkl_files); \
dataset_trainer(f, p, t)"
```
