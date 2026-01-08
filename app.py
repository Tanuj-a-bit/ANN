import os
import torch
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

from src.model import HandwritingModel
from src.utils import decode_prediction
from config import *

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Load Model
model = HandwritingModel(num_classes=NUM_CLASSES, hidden_size=HIDDEN_SIZE).to(DEVICE)
model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

if os.path.exists(model_path):
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print("Warning: Model weights not found. Predictions will be random.")
model.eval()

def preprocess_image(image_data):
    # Decode base64 image
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    
    # Resize to match model input
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Ensure ink is high value (1.0) and paper is low value (0.0)
    # The canvas drawing is black ink on white paper, so we need to invert it.
    if np.mean(img_np) > 0.5:
        img_np = 1.0 - img_np
        
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        input_tensor = preprocess_image(data['image'])
        with torch.no_grad():
            output = model(input_tensor)
            prediction = decode_prediction(output)[0]
            
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': os.path.exists(model_path),
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
