import torch
from torchvision import transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import os

from src.model import get_model
from src.utils import colorize_mask

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=19, backbone='resnet50') 
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Using random weights (for testing only).")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size[::-1]
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        prediction = output.argmax(0)

    prediction_pil = colorize_mask(prediction)
    prediction_pil = prediction_pil.resize(image.size, Image.NEAREST)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(prediction_pil)
    plt.title("Segmentation Prediction")
    plt.axis('off')
    
    if args.output:
        plt.savefig(args.output)
        print(f"Saved result to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None, help='Path to save output visualization')
    args = parser.parse_args()
    
    inference(args)
