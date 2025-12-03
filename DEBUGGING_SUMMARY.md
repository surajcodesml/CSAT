# CSAT Detector Training - Debugging Summary

## Issues Found and Fixed

### 1. Data Loading Issues (dataloader.py)

#### Problem 1: Incorrect Label Mapping
**Location:** `utils/dataloader.py` - `create_labelout()` method

**Original Code:**
```python
c = [cl+1 if cl <2 else 0 for cl in c]
if 0 in c:
    b = torch.empty((0,4))
    c = torch.zeros((0,1))
```

**Issue:** 
- Labels were incorrectly mapped: 0→1, 1→2, 2→0
- Then filtered label 0, which removed "no boxes" cases
- This inverted logic caused valid boxes to be filtered

**Fix:**
- Filter label 2 (which means "no boxes") correctly
- Keep labels 0 (Fovea) and 1 (SCR) as valid classes
- Add 1 to make them 1-indexed: 0→1 (Fovea), 1→2 (SCR)
- Also filter all-zero bounding boxes

**New Code:**
```python
# Filter out invalid boxes (label == 2 or all-zero boxes)
valid_mask = (c != 2) & ~np.all(b == 0, axis=1)
c_filtered = c[valid_mask]
b_filtered = b[valid_mask]

if len(c_filtered) == 0:
    # No valid boxes - return empty tensors
    b_tensor = torch.empty((0, 4))
    c_tensor = torch.zeros((0, 1))
else:
    # Add 1 to labels: 0→1 (Fovea), 1→2 (SCR)
    c_filtered = c_filtered + 1
    b_tensor = torch.from_numpy(b_filtered).float()
    c_tensor = torch.from_numpy(c_filtered).reshape((-1, 1)).float()
```

#### Problem 2: Tensor Type Handling
**Location:** `utils/dataloader.py` - `load_pickle()` method

**Issue:** 
- Pickle files contain torch.Tensors, not numpy arrays
- Code didn't handle tensor-to-numpy conversion
- This caused inconsistent data types downstream

**Fix:**
```python
# Ensure numpy arrays for consistent processing
if isinstance(im, torch.Tensor):
    im = im.numpy()
if isinstance(bo, torch.Tensor):
    bo = bo.numpy()
if isinstance(clss, torch.Tensor):
    clss = clss.numpy()
```

### 2. Pretrained Weights Loading Issues (train.py)

#### Problem 1: Inflexible Path Handling
**Location:** `train.py` - `load_saved_model()` function

**Issue:**
- Function only looked in `root + 'runs/' + weights_path` 
- Didn't support absolute paths or direct file paths
- Pretrained weights are in `model/best_pretrainer.pth`, not in `runs/`

**Fix:**
```python
# Handle both absolute paths and relative paths
if os.path.isabs(weights_path) or os.path.exists(weights_path):
    ckptfile = weights_path
elif os.path.exists(root + weights_path):
    ckptfile = root + weights_path
else:
    ckptfile = root + 'runs/' + weights_path + '.pth'

if not os.path.exists(ckptfile):
    raise FileNotFoundError(f"Checkpoint file not found: {ckptfile}")
```

#### Problem 2: Missing strict=False Parameter
**Issue:**
- `load_state_dict()` called without `strict=False`
- Would fail if encoder has extra keys not in pretrained weights

**Fix:**
```python
M.load_state_dict(new_state_dict, strict=False)
```

#### Problem 3: Wrong Default Path
**Location:** `train.py` - `arg_parse()` function

**Issue:**
- Default pretrain_weights was `'0best_pretrainer'` (incorrect)
- Default dataroot was `'./data'` (should be `'./pickle'`)

**Fix:**
```python
parser.add_argument('--pretrain_weights', type=str, default='model/best_pretrainer.pth', ...)
parser.add_argument('--dataroot', type=str, default='./pickle', ...)
```

### 3. Missing Data Split Files

#### Problem: Empty train.txt and val.txt
**Issue:**
- `data/train.txt` and `data/val.txt` were empty (0 bytes)
- Training would fail with no data

**Fix:**
- Generated train/val splits using `utils/util.py::dataset_trainer()`
- Result: 10,026 training samples, 2,597 validation samples

## Data Format Verification

### Pickle File Structure
Each pickle file contains:
- `'img'`: torch.Tensor of shape [3, 256, 576] (CHW format)
- `'box'`: torch.Tensor of shape [N, 4] (normalized [x_center, y_center, width, height])
- `'label'`: torch.Tensor of shape [N] where:
  - 0 = Fovea
  - 1 = SCR
  - 2 = No boxes (should be filtered)
- `'name'`: str (filename without extension)

### Expected Output Format
After processing:
- Images: `[r, C, H, W]` where r=3 (adjacent frames)
- Labels: `[N, 6]` where columns are `[batch_idx, class, x_center, y_center, width, height]`
- Classes: 1 (Fovea), 2 (SCR) after +1 offset

## Testing Results

All tests PASSED ✓

### Test 1: Data Loading
- ✓ Successfully loads pickle files
- ✓ Correctly filters label 2 (no boxes)
- ✓ Proper label mapping: 0→1, 1→2
- ✓ Bounding boxes remain normalized [0, 1]
- ✓ Image format: [3, 3, 256, 576] (r=3, C=3, H=256, W=576)

### Test 2: Pretrained Weights
- ✓ Loads checkpoint from model/best_pretrainer.pth
- ✓ Strips 'module.encoder.' prefix correctly
- ✓ Encoder forward pass works
- ✓ Full Dent_Pt model forward pass works
- ✓ Pretrained accuracy: 0.4980

## Recommendations

1. **Training Command:**
```bash
python train.py --pretrain True --epochs 100 --train_batch 64 --val_batch 64
```

2. **Monitor These Metrics:**
- Loss values should be visible in console now
- Check for proper class distribution in predictions
- Monitor False Negative (FN) rate specifically for Fovea and SCR classes

3. **If High FN Rate Persists:**
- Consider adjusting the number of queries (`num_queries`) in the decoder
- Review loss weights in `loss/loss_criterion.py`
- Check if class imbalance needs addressing (use focal loss parameters)

## Files Modified

1. `utils/dataloader.py`: Fixed label mapping and tensor handling
2. `train.py`: Fixed pretrained weights loading and default paths
3. `data/train.txt`: Generated (10,026 samples)
4. `data/val.txt`: Generated (2,597 samples)

## Files Created

1. `test_dataloader.py`: Comprehensive test script for data and weights
2. `DEBUGGING_SUMMARY.md`: This file
