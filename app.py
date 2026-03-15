import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- 1. Model Definition ---
# This must perfectly match the architecture used during training
def get_model(num_classes=7):
    model = models.convnext_tiny()
    # Adjust the head for our 7 diagnostic classes
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

# --- 2. Initialize and Load Weights ---
device = torch.device("cpu")
model = get_model()

# Path to your saved model weights
MODEL_PATH = "best_melanoma_recall_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Successfully loaded weights from {MODEL_PATH}")
else:
    print(f"❌ Error: {MODEL_PATH} not found in root directory!")

model.to(device)
model.eval()

# --- 3. Clinical Configuration ---
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
MELANOMA_SAFETY_THRESHOLD = 0.20  # Our custom clinical override

# --- 4. Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 5. Prediction Logic ---
def predict(img):
    if img is None:
        return None, "Please upload a dermoscopic image."
    
    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Extract Melanoma probability (Index 4)
    mel_prob = float(probs[4])
    
    # Apply Clinical Safety Override
    if mel_prob >= MELANOMA_SAFETY_THRESHOLD:
        pred_idx = 4
        clinical_status = (
            f"🚨 CLINICAL ALERT: High Risk Detected.\n"
            f"Melanoma Probability: {mel_prob:.2%}\n"
            f"This exceeds the safety threshold of {MELANOMA_SAFETY_THRESHOLD:.0%}. "
            "Flagging for urgent specialist review."
        )
    else:
        pred_idx = torch.argmax(probs).item()
        clinical_status = (
            f"✅ Routine Review: Primary prediction is {CLASSES[pred_idx]}.\n"
            f"Melanoma Probability: {mel_prob:.2%}.\n"
            "This is below the safety override threshold."
        )

    # Format result for Gradio Label
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, clinical_status

# --- 6. Build the Professional Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 DermAI: High-Sensitivity Clinical Screening")
    
    with gr.Accordion("📊 Clinical Model Audit Info", open=False):
        gr.Markdown("""
        **Performance Summary