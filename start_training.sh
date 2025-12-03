#!/bin/bash
# Quick start script for training the detector with pretrained weights

echo "Starting CSAT Detector Training"
echo "================================"
echo ""
echo "Configuration from config.yaml:"
echo "  - Pretrained weights: model/best_pretrainer.pth"
echo "  - Data path: ./pickle"
echo "  - Training samples: 10,026"
echo "  - Validation samples: 2,597"
echo "  - Epochs: 100"
echo "  - Batch size: 64"
echo ""

# Run training with config file
python train.py --config config.yaml

echo ""
echo "Training completed!"
