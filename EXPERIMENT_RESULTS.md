# Hyperparameter Experiment Results Summary

## Experiment Configurations

This document will be updated with results as experiments complete.

### Experiment 1: Baseline
- **Status**: Running
- **Config**: LR=0.0002, Batch=128, Queries=100, Dropout=0.1
- **Epochs**: 20
- **Purpose**: Establish baseline performance

### Experiment 2: Learning Rate Tuning
- **Status**: Pending
- **Config**: LR=0.00005 (4x lower), Batch=128, Queries=100
- **Changes**: Lower LR, gentler scheduler (patience=8, factor=0.7)
- **Hypothesis**: Current LR too high for transfer learning

### Experiment 3: Loss Weight Balancing
- **Status**: Pending
- **Config**: LR=0.0002, Batch=128, Queries=100
- **Changes**: Loss weights CE:2, BBox:5, GIoU:2 (vs current CE:5, BBox:3, GIoU:8)
- **Hypothesis**: More balanced loss weights reduce FP/FN

### Experiment 4: Architecture Tuning
- **Status**: Pending
- **Config**: LR=0.0002, Batch=128, Queries=50, Dropout=0.15
- **Changes**: Reduced queries from 100→50, increased dropout
- **Hypothesis**: Too many queries causes matching confusion

### Experiment 5: Batch Size Reduction
- **Status**: Pending
- **Config**: LR=0.0001, Batch=32, Queries=100
- **Changes**: Smaller batch for gradient diversity
- **Hypothesis**: Large batch reduces exploration

### Experiment 6: Combined Optimized
- **Status**: Pending
- **Config**: LR=0.00008, Batch=48, Queries=50, Dropout=0.15
- **Epochs**: 50 (longer run)
- **Changes**: Combined best insights from experiments 2-5

---

## Results

Results will be filled in automatically after running `compare_experiments.py`.

---

## Best Configuration

**TBD** - Will be determined after all experiments complete.

---

Last updated: Experiment setup complete, awaiting results.
