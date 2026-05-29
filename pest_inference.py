import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class PestCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Relative to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing_path(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_PATH = _first_existing_path([
    os.path.join(BASE_DIR, "..", "models", "custom_cnn_model.pth"),
    os.path.join(BASE_DIR, "models", "custom_cnn_model.pth"),
])
CLASS_NAMES_PATH = _first_existing_path([
    os.path.join(BASE_DIR, "..", "models", "class_names.json"),
    os.path.join(BASE_DIR, "models", "class_names.json"),
])
PEST_DETAILS_PATH = _first_existing_path([
    os.path.join(BASE_DIR, "..", "models", "pest_details.json"),
    os.path.join(BASE_DIR, "models", "pest_details.json"),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES: List[str] = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)


# Optional pest details
if os.path.exists(PEST_DETAILS_PATH):
    with open(PEST_DETAILS_PATH, "r", encoding="utf-8") as f:
        PEST_DATABASE: List[Dict[str, Any]] = json.load(f)
else:
    PEST_DATABASE = []


# Build model and load weights
model = PestCNN(NUM_CLASSES).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


# Inference transform (must match training/inference pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def get_pest_info(pest_name: str) -> Optional[Dict[str, Any]]:
    for pest in PEST_DATABASE:
        if str(pest.get("name", "")).lower() == pest_name.lower():
            return pest
    return None


def predict_pest(image_path: str) -> Tuple[str, float, Dict[str, Any]]:
    """
    Predict pest class from an uploaded image path.

    Returns:
        predicted_class: str
        confidence_percent: float
        pest_info: dict (empty if not found)
    """
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence_percent = conf.item() * 100.0

    pest_info = get_pest_info(predicted_class) or {}

    return predicted_class, confidence_percent, pest_info
