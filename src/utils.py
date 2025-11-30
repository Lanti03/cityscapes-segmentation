import torch
import numpy as np

def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
            
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return ious, mean_iou

CITYSCAPES_PALETTE = [
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],    # building
    [102, 102, 156], # wall
    [190, 153, 153], # fence
    [153, 153, 153], # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],   # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152], # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],   # person
    [255, 0, 0],     # rider
    [0, 0, 142],     # car
    [0, 0, 70],      # truck
    [0, 60, 100],    # bus
    [0, 80, 100],    # train
    [0, 0, 230],     # motorcycle
    [119, 11, 32],   # bicycle
    [0, 0, 0]        # void/background
]

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def colorize_mask(mask):
    from PIL import Image
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label_id, color in enumerate(CITYSCAPES_PALETTE):
        if label_id >= 19:
             continue
        color_mask[mask == label_id] = color
        
    return Image.fromarray(color_mask)
