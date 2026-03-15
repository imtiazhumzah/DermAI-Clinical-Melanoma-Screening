import gradio as gr
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
from datetime import datetime

# Import your architecture and logic from the local file
from model_utils import get_model, get_transforms, apply_clinical_logic

# --- 1. Setup & Configuration ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_melanoma_recall_model.pth"
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

model = get_model(num_classes=7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = get_transforms()

# --- 2. Explainability: Grad-CAM for ConvNeXt ---
def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    # For ConvNeXt Tiny, block 7 is the final stage before global pooling
    target_layer = model.features[7] 
    
    activations = []
    gradients = []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward/Backward
    logits = model(img_tensor)
    model.zero_grad()
    score = logits[0, target_class_idx]
    score.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    
    # ConvNeXt-Tiny stage 7 output is (B, C, H, W) or (B, H, W, C) depending on version
    # We'll calculate weights based on spatial dimensions
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze().cpu().numpy()
    
    # Process heatmap
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(original_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    h1.remove()
    h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 3. Reporting Logic ---
from fpdf import XPos, YPos

# --- 3. Updated Reporting Logic (fpdf2 2.7.8+ compatible) ---
# --- 3. Updated Reporting Logic with Writable Path ---
def generate_report(image, diagnosis, confidence, age, sex, site, status):
    clean_status = status.replace("🚨 ", "").replace("✅ ", "")
    
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, text="DermAI Clinical Screening Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, text=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, 30, 200, 30)
    
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Patient Info:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 10, text=f"Age: {age} | Sex: {sex} | Site: {site}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Results:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 10, text=f"Primary Diagnosis: {diagnosis} ({confidence})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 10, text=f"Status: {clean_status}")
    
    # FIX: Use /tmp/ to bypass Permission Denied errors in Docker
    report_path = "/tmp/clinical_report.pdf" 
    pdf.output(report_path)
    return report_path

# --- 4. Prediction Wrapper (Updated to prevent UserWarning) ---
def predict_clinical(img, age, sex, site):
    if img is None: return None, "Upload image.", None, None
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True
    
    logits = model(img_tensor)
    # Using .detach() here prevents the PyTorch UserWarning
    probs = torch.softmax(logits, dim=1)[0].detach() 
    
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES)
    diag = CLASSES[pred_idx]
    conf_str = f"{probs[pred_idx]:.2%}"
    
    if pred_idx == 4:
        status = "🚨 HIGH RISK: Melanoma safety threshold triggered."
    else:
        status = f"✅ Routine Review: Melanoma probability ({mel_prob:.2%}) low."

    # Need the tensor with grad for Grad-CAM, so we pass it separately or re-run
    heatmap = generate_gradcam(model, img_tensor, img, pred_idx)
    report = generate_report(img, diag, conf_str, age, sex, site, status)
    
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, status, report, heatmap

# --- 5. UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Advanced Clinical Screening")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Dermoscopic Input")
            with gr.Row():
                age_in = gr.Number(label="Age", value=45)
                sex_in = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
            site_in = gr.Dropdown(["Face", "Back", "Chest", "Extremity", "Other"], label="Site", value="Back")
            run_btn = gr.Button("Analyze & Generate Report", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(label="Diagnostic Probabilities")
            output_heatmap = gr.Image(label="Explainability (Grad-CAM)")
            output_status = gr.Textbox(label="Clinical Recommendation")
            output_pdf = gr.File(label="Download PDF Report")

    run_btn.click(
        fn=predict_clinical, 
        inputs=[input_img, age_in, sex_in, site_in], 
        outputs=[output_labels, output_status, output_pdf, output_heatmap]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())