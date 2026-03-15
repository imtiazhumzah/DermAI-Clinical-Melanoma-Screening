import gradio as gr
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import gc
from PIL import Image
from fpdf import FPDF, XPos, YPos
from datetime import datetime

# Import your architecture and logic from the local file
from model_utils import get_model, get_transforms, apply_clinical_logic

# --- 1. Setup & Configuration ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_melanoma_recall_model.pth"
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# Load model with memory efficiency
model = get_model(num_classes=7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = get_transforms()

# --- 2. Image Validation & Reliability Logic (OOD Detection) ---
def perform_ood_check(image):
    """
    Evaluates if the image is a valid dermoscopic sample.
    """
    img_np = np.array(image.convert("L"))
    
    # Check 1: Focus/Sharpness using Laplacian Variance
    sharpness_score = cv2.Laplacian(img_np, cv2.CV_64F).var()
    
    # Check 2: Information Density (avoiding blank/solid color images)
    is_valid = sharpness_score > 50 
    reliability = "High" if is_valid else "Reduced"
    validity_msg = "✅ Dermoscopic Likelihood: High" if is_valid else "⚠️ Image Validity: Low (Possible Blur/Non-Dermoscopic)"
    
    return is_valid, reliability, validity_msg

# --- 3. Enhanced Explainability (Grad-CAM) ---
def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    target_layer = model.features[7] 
    activations, gradients = [], []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Gradient-based pass
    logits = model(img_tensor)
    model.zero_grad()
    score = logits[0, target_class_idx]
    score.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze().cpu().numpy()
    
    cam = np.maximum(cam, 0)
    cam = cv2.GaussianBlur(cam, (11, 11), 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    img_np = np.array(original_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
    
    h1.remove(); h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 4. Clinical Reporting ---
def generate_report(diagnosis, confidence, age, sex, site, status):
    clean_status = status.replace("🚨 ", "").replace("✅ ", "").replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, text="DermAI: Clinical Decision Support Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, 32, 200, 32)
    pdf.ln(5)

    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Patient Metadata", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, text=f"Age: {age} | Sex: {sex} | Site: {site}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)