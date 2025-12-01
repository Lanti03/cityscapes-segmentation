import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import os
import numpy as np
import csv

from src.dataset import CityscapesDataset
from src.model import get_model
from src.utils import calculate_iou, CITYSCAPES_CLASSES

def train(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    target_transform = None

    NUM_CLASSES = 19

    if not os.path.exists(args.data_path):
        print(f"Data path {args.data_path} does not exist. Please download Cityscapes.")
        return

    # Stable DataLoader settings for Mac M-series
    # workers=0 avoids multiprocessing overhead/deadlocks
    # pin_memory=False because it's not supported on MPS yet
    num_workers = args.workers
    
    train_dataset = CityscapesDataset(args.data_path, split='train', transform=transform, target_transform=target_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_dataset = CityscapesDataset(args.data_path, split='val', transform=transform, target_transform=target_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    model = get_model(num_classes=NUM_CLASSES, backbone='resnet50').to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_miou = checkpoint['best_miou']
            else:
                model.load_state_dict(checkpoint)
                start_epoch = int(args.resume.split('_')[-1].replace('.pth', ''))
                
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    os.makedirs('logs', exist_ok=True)
    csv_file = 'logs/training_metrics.csv'
    if start_epoch == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'mIoU', 'Best_mIoU'])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0.0
        total_ious = []
        
        print(f"Epoch {epoch+1} [Validation]...")
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                ious, _ = calculate_iou(preds, targets, NUM_CLASSES)
                total_ious.append(ious)
        
        total_ious = np.array(total_ious)
        mean_class_iou = np.nanmean(total_ious, axis=0)
        miou = np.nanmean(mean_class_iou)
        
        print(f"Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Epoch {epoch+1} mIoU: {miou:.4f}")
        print("-" * 30)
        print("Class-wise IoU:")
        for cls_name, iou in zip(CITYSCAPES_CLASSES, mean_class_iou):
            print(f"{cls_name:15s}: {iou:.4f}")
        print("-" * 30)

        # Log metrics to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, running_loss / len(train_loader), val_loss / len(val_loader), miou, best_miou])

        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
        
        if miou > best_miou:
            best_miou = miou
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou
            }
            torch.save(best_checkpoint, "checkpoints/best_model.pth")
            print(f"New best model saved with mIoU: {miou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to Cityscapes dataset root')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    args = parser.parse_args()
    
    train(args)
