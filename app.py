import gradio as gr
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF, XPos, YPos
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

# --- 2. Enhanced Explainability (Smooth Heatmaps) ---
def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    target_layer = model.features[7] 
    activations, gradients = [], []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    logits = model(img_tensor)
    model.zero_grad()
    score = logits[0, target_class_idx]
    score.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze().cpu().numpy()
    
    cam = np.maximum(cam, 0)
    # SMOOTHING: Apply Gaussian Blur for a more professional look
    cam = cv2.GaussianBlur(cam, (11, 11), 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # COLORMAP: Using COLORMAP_HOT for better clinical contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    img_np = np.array(original_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
    
    h1.remove(); h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 3. Clinical Reporting (No Emojis/Modern Syntax) ---
def generate_report(diagnosis, confidence, age, sex, site, status):
    clean_status = status.replace("🚨 ", "").replace("✅ ", "")
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
    pdf.cell(0, 8, text=f"Age: {age} | Sex: {sex} | Anatomical Site: {site}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Automated Diagnostic Findings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, text=f"Primary Prediction: {diagnosis} (Confidence: {confidence})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 8, text=f"Safety Status: {clean_status}")
    
    report_path = f"/tmp/DermAI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(report_path)
    return report_path

# --- 4. Logic Wrapper ---
def predict_clinical(img, age, sex, site):
    if img is None: return None, "Please upload an image.", None, None
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True
    
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0].detach()
    
    # Clinical Safety Logic (20% Threshold for Melanoma)
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES)
    diag = CLASSES[pred_idx]
    conf_str = f"{probs[pred_idx]:.2%}"
    
    if pred_idx == 4 or mel_prob > 0.20:
        status = "🚨 HIGH RISK: Safety threshold triggered for Melanoma."
    else:
        status = "✅ Routine Review: Low risk of malignancy detected."

    heatmap = generate_gradcam(model, img_tensor, img, pred_idx)
    report = generate_report(diag, conf_str, age, sex, site, status)
    
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, status, report, heatmap

# --- 5. UI Construction (Side-by-Side Layout) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🩺 DermAI: Advanced Clinical Screening Workflow")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(type="pil", label="1. Upload Dermoscopic Image")
            with gr.Row():
                age_in = gr.Number(label="Age", value=45)
                sex_in = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
            site_in = gr.Dropdown(["Face", "Back", "Chest", "Extremity", "Other"], label="Site", value="Back")
            run_btn = gr.Button("🚀 Perform Clinical Audit", variant="primary")
        
        with gr.Column(scale=3):
            with gr.Row():
                # Side-by-Side comparison of original and explainability
                input_preview = gr.Image(label="Original View", interactive=False)
                output_heatmap = gr.Image(label="Explainability (Grad-CAM)")
            
            output_labels = gr.Label(label="Diagnostic Confidence")
            output_status = gr.Textbox(label="Clinical Recommendation", interactive=False)
            output_pdf = gr.File(label="Download Full Clinical Report")

    # Sync input image to the preview slot automatically
    input_img.change(fn=lambda x: x, inputs=input_img, outputs=input_preview)

    run_btn.click(
        fn=predict_clinical, 
        inputs=[input_img, age_in, sex_in, site_in], 
        outputs=[output_labels, output_status, output_pdf, output_heatmap]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)