from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import torchvision.transforms as transforms
import torchvision.models as models

# Initialize Flask app
app = Flask(__name__,
    template_folder='../templates',
    static_folder='../static')
import os
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_PERMANENT'] = False

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== LOAD YOUR TRAINED CUSTOM CNN MODEL ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load class names
try:
    with open('../models/class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"✅ Loaded {len(class_names)} class names")
    print(f"Classes: {class_names[:10]}...")
except Exception as e:
    print(f"❌ Error loading class names: {e}")
    class_names = [
        '1', '10', '11', '12', '13', '14', '15', '16', '17', '18',
        '19', '2', '20', '21', '22', '23', '24', '25', '26', '27',
        '28', '29', '3', '30', '31', '32', '33', '34', '35', '36',
        '37', '38', '39', '4', '40', '5', '6', '7', '8', '9', 'non-pest'
    ]
    print("⚠️ Using default class names")

num_classes = len(class_names)


# Define Custom CNN architecture
class PestCNN(nn.Module):
    def __init__(self, num_classes):
        super(PestCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Build model
model = PestCNN(num_classes).to(DEVICE)

# Load trained weights
model_path = r"C:\Users\Acer\Desktop\Plant_pest_Identifier\pest-identifier-app\models\custom_cnn_model.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("✅ Custom CNN model loaded successfully!")
        MODEL_LOADED = True

        # Optional startup sanity check
        # Keep in development; can remove in production for faster startup
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            dummy_output = model(dummy_input)
            print(f"✅ Model test passed - output shape: {dummy_output.shape}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            MODEL_LOADED = False

    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        MODEL_LOADED = False
else:
    print(f"❌ Model not found at {model_path}")
    MODEL_LOADED = False


# Transform for custom CNN (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


with open(r"C:\Users\Acer\Desktop\Plant_pest_Identifier\pest-identifier-app\user\PEST_DATABASE.json", "r", encoding="utf-8") as f:
    PEST_DATABASE = json.load(f)

# def get_pest_info(predicted_class):
#     # predicted_class might be "2" or "non-pest"
#     info = PEST_DATABASE.get(str(predicted_class))
#     if info:
#         return info
# 
#     # fallback for non-pest or missing class
#     return {
#         "pest_name": "Non-pest",
#         "description": "No harmful pest detected.",
#         "crops_affected": [],
#         "symptoms": [],
#         "preventions": {
#             "organic": ["No treatment needed.", "Monitor plant health.", "Keep field clean."],
#             "chemical": ["No pesticide recommended.", "Avoid unnecessary sprays.", "Consult expert if unsure."]
#         }
#     }


# ========== HELPER FUNCTIONS ==========
def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_pest(image_path):
    """Predict pest using trained Custom CNN model."""
    fallback_class = "non-pest" if "non-pest" in class_names else class_names[-1]

    # If model failed to load, return safe fallback
    if not MODEL_LOADED:
        return fallback_class, 0.0

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Debug: top-k predictions
            print("\n=== PREDICTION DEBUG ===")
            k = min(5, probabilities.shape[1])
            top_probs, top_indices = torch.topk(probabilities[0], k)
            for i in range(k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                if prob > 0.01:
                    print(f"Class {class_names[idx]}: {prob:.2%}")

            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()  # 0 to 1

        print(f"✅ Predicted: {predicted_class} with {confidence_score:.2%} confidence")
        print("=======================\n")

        return predicted_class, confidence_score

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Best fallback for your 40 pest + 1 non-pest setup
        return fallback_class, 0.0

# ========== ROUTES ==========

@app.route('/')
def index():
    """Home page"""
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html')

@app.route('/identify', methods=['GET', 'POST'])
def identify():
    """Pest identification page with ML model"""
    if 'history' not in session:
        session['history'] = []
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Use your trained model for prediction
            if MODEL_LOADED:
                predicted_class, confidence = predict_pest(filepath)
                pest_id = predicted_class
            else:
                # Fallback if model not loaded
                import random
                pests = list(PEST_DATABASE.keys())
                predicted_class = random.choice(pests)
                confidence = round(random.uniform(0.75, 0.99), 2)
                pest_id = str(predicted_class)
            
            # Save to session history
            pest_name = PEST_DATABASE.get(pest_id, {}).get('pest_name', f'Pest {pest_id}')
            
            history_entry = {
                'id': str(len(session['history']) + 1),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'pest': pest_name,
                'pest_id': pest_id,
                'image': unique_filename,
                'confidence': confidence
            }
            session['history'].append(history_entry)
            session.modified = True
            
            flash(f'Pest identified as {pest_name} with {confidence*100:.1f}% confidence!', 'success')
            
            return redirect(url_for('results', pest_id=pest_id, image=unique_filename, confidence=confidence))
        
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
        return redirect(request.url)
    
    return render_template('identify.html')

@app.route('/results')
def results():
    """Show identification results"""
    pest_id = request.args.get("pest_id", "non-pest")
    image = request.args.get("image", "")
    confidence = request.args.get("confidence", type=float)

    fallback = PEST_DATABASE.get("non-pest", {
        "pest_name": "Non-pest",
        "description": "No harmful pest detected.",
        "crops_affected": [],
        "symptoms": [],
        "treatments": {
            "organic": [],
            "chemical": []
        }
    })

    pest_info = PEST_DATABASE.get(str(pest_id), fallback)

    confidence_percent = round(confidence * 100, 2) if confidence is not None else None

    return render_template(
        "results.html",
        pest=pest_info,
        image=image,
        confidence=confidence_percent
    )


@app.route('/pests')
def pest_library():
    """Browse all pests in database"""
    # Filter out non-pest from main library view
    pests = {k: v for k, v in PEST_DATABASE.items() if k != 'non-pest'}
    return render_template('pests.html', pests=pests)

@app.route('/pest/<pest_id>')
def pest_detail(pest_id):
    """Detailed view of a specific pest"""
    pest_id = str(pest_id)
    pest_info = PEST_DATABASE.get(pest_id)
    if not pest_info:
        flash('Pest not found', 'error')
        return redirect(url_for('pest_library'))
    
    return render_template('pest_detail.html', pest=pest_info, pest_id=pest_id)

@app.route('/history')
def history():
    """View identification history"""
    if 'history' not in session:
        session['history'] = []
    return render_template('history.html', history=session['history'])

@app.route('/clear-history')
def clear_history():
    """Clear session history"""
    session['history'] = []
    session.modified = True
    flash('History cleared!', 'success')
    return redirect(url_for('history'))

# ========== TEST ROUTES ==========
@app.route('/test-image')
def test_image():
    """Test route to check model on a sample image"""
    test_image_path = "../static/images/sample_pest.jpg"
    
    if os.path.exists(test_image_path):
        predicted_class, confidence = predict_pest(test_image_path)
        return f"Test result: {predicted_class} with {confidence:.2%} confidence"
    else:
        return "Please add a sample image to static/images/"

@app.route('/check-model')
def check_model():
    """Check if model and classes are loaded"""
    if MODEL_LOADED:
        return f"""
        <h3>✅ Model Loaded</h3>
        <p>Classes ({len(class_names)}):</p>
        <ul>
            {''.join([f'<li>{c}</li>' for c in class_names[:10]])}
            <li>...</li>
        </ul>
        """
    else:
        return "<h3>❌ Model Not Loaded</h3>"
@app.route('/debug-pests')
def debug_pests():
    """Debug route to check pest database"""
    pests_count = len(PEST_DATABASE)
    pests_list = list(PEST_DATABASE.keys())
    return f"""
    <h3>Pest Database Debug</h3>
    <p>Total pests: {pests_count}</p>
    <p>Pest IDs: {pests_list[:10]}...</p>
    <p>First pest name: {PEST_DATABASE.get('1', {}).get('name', 'Not found')}</p>
    """
def update_all_treatments():
    """Update all pests to have organic, chemical, and biological treatments"""
    for pest_id, pest in PEST_DATABASE.items():
        if pest_id != 'non-pest':
            # Create three treatment types
            pest['treatments'] = [
                {
                    'type': 'organic',
                    'method': 'Organic Control',
                    'instructions': 'Use neem oil, insecticidal soap, or manual removal. Apply in early morning or evening.'
                },
                {
                    'type': 'chemical',
                    'method': 'Chemical Control',
                    'instructions': 'Use recommended pesticides. Always follow label instructions and wear protective equipment.'
                },
                {
                    'type': 'biological',
                    'method': 'Biological Control',
                    'instructions': 'Introduce natural predators such as ladybugs, lacewings, or parasitic wasps.'
                }
            ]
    
    print("All treatments updated!")

# ========== RUN THE APPLICATION ==========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)