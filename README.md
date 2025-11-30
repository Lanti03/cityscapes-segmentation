# Road Segmentation Project
# Road Segmentation - Cityscapes

Semantic segmentation for autonomous vehicles using DeepLabV3+ on the Cityscapes dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
PYTHONPATH=. python src/train.py --data_path data --epochs 50 --batch_size 4
```

## Inference

```bash
PYTHONPATH=. python src/inference.py --image_path image.png --checkpoint checkpoints/best_model.pth --output result.png
```
# cityscapes-segmentation
