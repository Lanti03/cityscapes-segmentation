import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_dataset(source_root, target_root, size=(1024, 512)):
    """
    Resizes images and masks from source_root to target_root.
    """
    print(f"Resizing dataset from {source_root} to {target_root} with size {size}...")
    
    # Walk through source directory
    for root, dirs, files in os.walk(source_root):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith('.png') or file.endswith('.jpg'):
                # Construct paths
                rel_path = os.path.relpath(root, source_root)
                target_dir = os.path.join(target_root, rel_path)
                
                os.makedirs(target_dir, exist_ok=True)
                
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                
                try:
                    img = Image.open(source_path)
                    
                    if 'gtFine' in source_root:
                        img = img.resize(size, Image.NEAREST)
                    else:
                        img = img.resize(size, Image.BILINEAR)
                        
                    img.save(target_path)
                except Exception as e:
                    print(f"Error processing {source_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help='Original data root')
    parser.add_argument('--output_path', type=str, default='data_small', help='Output data root')
    args = parser.parse_args()
    
    # Resize images
    resize_dataset(
        os.path.join(args.data_path, 'leftImg8bit'), 
        os.path.join(args.output_path, 'leftImg8bit')
    )
    
    # Resize masks
    resize_dataset(
        os.path.join(args.data_path, 'gtFine'), 
        os.path.join(args.output_path, 'gtFine')
    )
    
    print("Done! Now run training with --data_path data_small")
