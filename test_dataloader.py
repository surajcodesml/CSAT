"""Test script to verify data loading and pretrained weights."""
import os
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Import custom modules
from utils.util import read_data
from utils.dataloader import create_dataloader
from model.transformer import Encoder, Dent_Pt

def test_data_loading():
    """Test that data is being loaded correctly."""
    print("=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)
    
    dataroot = './pickle'
    batch_size = 4
    r = 3
    space = 1
    
    # Get training data
    try:
        data = read_data('train')
        print(f"✓ Successfully read training data file")
        print(f"  Total samples: {len(data)}")
        print(f"  First 3 samples:")
        for i, sample in enumerate(data[:3]):
            print(f"    {i+1}. {sample}")
    except Exception as e:
        print(f"✗ Failed to read training data: {e}")
        return False
    
    # Create dataset
    try:
        dataset = create_dataloader(
            data[:100],  # Use subset for testing
            dataroot,
            batch_size,
            rank=-1,
            cache='ram',
            workers=0,
            phase='train',
            shuffle=False,
            r=r,
            space=space
        )
        print(f"\n✓ Successfully created dataset")
        print(f"  Dataset length: {len(dataset)}")
    except Exception as e:
        print(f"\n✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading a few batches
    print(f"\n  Testing batch loading...")
    try:
        for i in range(min(3, len(dataset))):
            images, labels, names = dataset[i]
            print(f"\n  Sample {i+1}:")
            print(f"    Images shape: {images.shape}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Labels: {labels}")
            print(f"    Name: {names}")
            
            # Verify data format
            assert images.dim() == 4, f"Expected 4D images [r, C, H, W], got {images.shape}"
            assert labels.dim() == 2, f"Expected 2D labels [N, 6], got {labels.shape}"
            
            if labels.shape[0] > 0:
                # Check label values
                classes = labels[:, 1]
                boxes = labels[:, 2:]
                print(f"    Classes: {classes}")
                print(f"    Boxes: {boxes}")
                
                # Verify classes are 1 or 2 (after mapping)
                assert torch.all((classes == 1) | (classes == 2)), \
                    f"Classes should be 1 or 2, got {classes}"
                
                # Verify boxes are normalized [0, 1]
                assert torch.all((boxes >= 0) & (boxes <= 1)), \
                    f"Boxes should be normalized [0, 1], got {boxes}"
                    
                print(f"    ✓ Data format is correct")
            else:
                print(f"    (No boxes in this sample)")
                
    except Exception as e:
        print(f"\n✗ Failed to load batches: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Data loading test PASSED")
    return True


def test_pretrained_weights():
    """Test that pretrained weights can be loaded correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Pretrained Weights Loading")
    print("=" * 80)
    
    pretrain_path = 'model/best_pretrainer.pth'
    
    # Check if file exists
    if not os.path.exists(pretrain_path):
        print(f"✗ Pretrained weights file not found: {pretrain_path}")
        return False
    
    print(f"✓ Pretrained weights file found: {pretrain_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        print(f"✓ Successfully loaded checkpoint")
        print(f"  Keys: {list(checkpoint.keys())}")
        
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_val_acc' in checkpoint:
            print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
            
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
    
    # Create encoder
    try:
        encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
        print(f"\n✓ Successfully created Encoder")
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\n✗ Failed to create encoder: {e}")
        return False
    
    # Load pretrained weights
    try:
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.encoder.' prefix
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace('module.encoder.', '')
            new_state_dict[new_key] = value
        
        # Load weights
        missing_keys, unexpected_keys = encoder.load_state_dict(new_state_dict, strict=False)
        
        print(f"\n✓ Successfully loaded pretrained weights into encoder")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"    {missing_keys[:5]}..." if len(missing_keys) > 5 else f"    {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"    {unexpected_keys}")
            
    except Exception as e:
        print(f"\n✗ Failed to load pretrained weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test encoder forward pass
    try:
        encoder.eval()
        with torch.no_grad():
            # The encoder processes each frame independently
            # Input: [batch * seq_len, C, H, W]
            dummy_input = torch.randn(3, 3, 256, 576)  # batch=3, C=3, H=256, W=576
            output = encoder(dummy_input)
            print(f"\n✓ Encoder forward pass successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
    except Exception as e:
        print(f"\n✗ Encoder forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full model
    try:
        model = Dent_Pt(encoder, hidden_dim=256, num_class=2)
        print(f"\n✓ Successfully created Dent_Pt model")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(3, 1, 3, 256, 576)  # [seq_len, batch, C, H, W]
            outputs = model(dummy_input)
            logits, boxes = outputs
            print(f"\n✓ Model forward pass successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Boxes shape: {boxes.shape}")
            
    except Exception as e:
        print(f"\n✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Pretrained weights test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING DATA LOADING AND PRETRAINED WEIGHTS")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test 1: Data loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test 2: Pretrained weights
    results.append(("Pretrained Weights", test_pretrained_weights()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + ("=" * 80))
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
