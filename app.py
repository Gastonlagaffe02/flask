from flask import Flask, render_template, request, redirect, url_for
import torch
import timm
from PIL import Image
import os
from utils import transform_image, predict_health, predict_disease

app = Flask(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = './static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ViT Models
# First model: Healthy vs Sick classification
health_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
health_model.load_state_dict(torch.load("model/vit_fish_disease.pth", map_location=torch.device('cpu')))
health_model.eval()

# Second model: Disease type classification
# Assuming the same architecture, but with 7 output classes
disease_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=6)
disease_model.load_state_dict(torch.load("model/classe.pth", map_location=torch.device('cpu')))
disease_model.eval()

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
health_model.to(device)
disease_model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the image to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # First, predict if the fish is healthy or sick
            health_status = predict_health(file_path, health_model, device)
            
            # If the fish is sick, predict the specific disease
            if health_status == "Sick":
                disease_type = predict_disease(file_path, disease_model, device)
                prediction = f"Sick - {disease_type}"
            else:
                prediction = "Healthy"
                
            # For API access, return JSON instead of HTML
            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'prediction': prediction,
                    'image_path': file_path
                })
            else:
                # Return HTML for web browser access
                return render_template('result.html', prediction=prediction, image_path=file_path)
    
    # For API access, return error as JSON
    if request.headers.get('Accept') == 'application/json':
        return jsonify({'error': 'No file uploaded'}), 400
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)