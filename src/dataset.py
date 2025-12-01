import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, 'gtFine', split)
        
        self.images = []
        self.targets = []
        
        # Create lookup table for fast mapping
        self.lookup_table = np.zeros(256, dtype=np.uint8)
        self.lookup_table[:] = 255 # Default to ignore index
        
        mapping = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
            7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
            14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
            21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
        }
        
        for k, v in mapping.items():
            if k >= 0:
                self.lookup_table[k] = v

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            
            if not os.path.isdir(img_dir):
                continue
                
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    self.targets.append(os.path.join(target_dir, target_name))

    def _encode_target(self, target):
        target = np.array(target)
        # Use lookup table for fast mapping
        mask = self.lookup_table[target]
        return torch.as_tensor(mask, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        
        # Transform image (ToTensor + Normalize)
        if self.transform:
            image = self.transform(image)
            
        # Map target labels and convert to tensor directly
        target = self._encode_target(target)

        return image, target
