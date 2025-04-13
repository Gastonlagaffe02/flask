from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
import timm
from PIL import Image
import os
from utils import transform_image, predict_health, predict_disease
import gdown

def download_model_if_needed(model_path, drive_id):
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, model_path, quiet=False)

# Setup
app = Flask(__name__)

# Uploads folder
UPLOAD_FOLDER = './static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model folder
MODEL_FOLDER = 'model'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# File paths
health_model_path = os.path.join(MODEL_FOLDER, 'vit_fish_disease.pth')
disease_model_path = os.path.join(MODEL_FOLDER, 'classe.pth')

# ✅ Download models from Google Drive if needed
download_model_if_needed(health_model_path, '1O6zx068_RdRDxLCqWdsr339UPpsB9F5F')
download_model_if_needed(disease_model_path, '11NQr_bJQ1GFDTEp4GH-Yp5caRJfcsNR4')

# Load models
health_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
health_model.load_state_dict(torch.load(health_model_path, map_location=torch.device('cpu')))
health_model.eval()

disease_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=6)
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
health_model.to(device)
disease_model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            health_status = predict_health(file_path, health_model, device)

            if health_status == "Sick":
                disease_type = predict_disease(file_path, disease_model, device)
                prediction = f"Sick - {disease_type}"
            else:
                prediction = "Healthy"

            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'prediction': prediction,
                    'image_path': file_path
                })
            else:
                return render_template('result.html', prediction=prediction, image_path=file_path)

    if request.headers.get('Accept') == 'application/json':
        return jsonify({'error': 'No file uploaded'}), 400
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var if available (e.g., Render)
    app.run(host='0.0.0.0', port=port, debug=True)

