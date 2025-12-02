import os
import io
import base64
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
from src.model import get_model
from src.utils import colorize_mask, CITYSCAPES_CLASSES

app = Flask(__name__)

# Configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
CHECKPOINT_PATH = 'checkpoints/checkpoint_epoch_2.pth'

# Load Model
print(f"Loading model on {DEVICE}...")
model = get_model(num_classes=NUM_CLASSES, backbone='resnet50')
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully!")
else:
    print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found. Using random weights.")

model.to(DEVICE)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    # Prepare legend data
    from src.utils import CITYSCAPES_PALETTE
    legend_data = []
    for i, name in enumerate(CITYSCAPES_CLASSES):
        if i >= 19: # Only use the 19 training classes
            break
        legend_data.append({
            'name': name,
            'color': CITYSCAPES_PALETTE[i]
        })
    return render_template('index.html', legend=legend_data)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        original_size = image.size
        
        # Resize for inference (optional, but good for speed/memory)
        # We'll use the same size as training for consistency, or keep original if powerful enough
        # Let's keep it somewhat standard
        input_image = image.resize((1024, 512))
        
        input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()

        # Colorize
        colorized_mask = colorize_mask(output_predictions)
        
        # Resize mask back to original image size
        colorized_mask = colorized_mask.resize(original_size, Image.NEAREST)

        # Convert to base64 for display
        buffered = io.BytesIO()
        colorized_mask.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'mask_image': f"data:image/png;base64,{mask_base64}"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
