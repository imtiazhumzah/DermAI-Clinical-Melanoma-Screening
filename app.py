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

model = get_model(num_classes=7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = get_transforms()

# --- 2. Advanced ABCDE & OOD Logic ---
def perform_ood_check(image):
    img_np = np.array(image.convert("L"))
    sharpness_score = cv2.Laplacian(img_np, cv2.CV_64F).var()
    is_valid = sharpness_score > 50 
    reliability = "High" if is_valid else "Reduced"
    validity_msg = "Dermoscopic Likelihood: High" if is_valid else "Image Validity: Low (Possible Blur/Non-Dermoscopic)"
    return is_valid, reliability, validity_msg

def calculate_abcde(img_cv, mask):
    """
    Computes A, B, C, and D indicators from the lesion and its mask.
    """
    # 1. Asymmetry (A)
    flipped_h = cv2.flip(mask, 1)
    asymmetry_score = np.sum(cv2.absdiff(mask, flipped_h)) / (np.sum(mask) + 1e-8)
    
    # 2. Border Irregularity (B) & 4. Diameter (D)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness = 0
    diameter_px = 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 0:
            compactness = (perimeter**2) / (4 * np.pi * area)
        _, _, w_box, h_box = cv2.boundingRect(cnt)
        diameter_px = max(w_box, h_box)

    # 3. Color Variation (C)
    pixels = img_cv[mask > 0]
    color_std = np.std(pixels, axis=0).mean() if pixels.size > 0 else 0

    return {
        "asymmetry": "High" if asymmetry_score > 0.3 else "Low",
        "border": "Irregular" if compactness > 1.6 else "Regular",
        "color": "Polychromatic" if color_std > 35 else "Monochromatic",
        "diameter": f"~{diameter_px * 0.1:.1f} mm" if diameter_px > 0 else "N/A"
    }

# --- 3. Enhanced Explainability (Grad-CAM) ---
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
    cam = cv2.GaussianBlur(cam, (11, 11), 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    img_np = np.array(original_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
    h1.remove(); h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 4. Clinical Reporting (Emoji-Safe) ---
def generate_report(diagnosis, confidence, age, sex, site, status_text, abcde):
    # Strip emojis for PDF compatibility
    clean_status = status_text.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "").replace("✅ ", "").replace("⚠️ ", "").replace("📊 ", "")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, text="DermAI: Clinical Decision Support Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, 32, 200, 32); pdf.ln(5)

    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Dermatological Indicators (ABCDE Analysis)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, text=f"Asymmetry: {abcde['asymmetry']} | Border: {abcde['border']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, text=f"Color: {abcde['color']} | Diameter: {abcde['diameter']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, text="Automated Diagnostic Findings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=11)
    pdf.cell(0, 8, text=f"Prediction: {diagnosis} ({confidence})", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 8, text=f"Clinical Guidance:\n{clean_status}")
    
    report_path = f"/tmp/DermAI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(report_path)
    return report_path

# --- 5. Main Clinical Pipeline ---
def predict_clinical(img, age, sex, site):
    if img is None: return [None] * 5
    gc.collect()
    
    # 1. Validation & Segmentation
    is_valid, reliability, validity_msg = perform_ood_check(img)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. ABCDE ANALYSIS
    abcde = calculate_abcde(img_cv, mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_overlay = img_cv.copy()
    cv2.drawContours(seg_overlay, contours, -1, (0, 255, 0), 2)
    segmentation_view = cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)

    # 3. Classification
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach()
    
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES)
    diag = CLASSES[pred_idx]
    conf_str = f"{probs[pred_idx]:.2%}"
    
    # 4. Actionable Triage Mapping
    if pred_idx == 4 or mel_prob > 0.20:
        triage = "🔴 HIGH RISK"
        action = "Urgent dermatology referral recommended."
    elif mel_prob > 0.10:
        triage = "🟡 MODERATE RISK"
        action = "Clinical correlation required."
    else:
        triage = "🟢 LOW RISK"
        action = "Routine clinical monitoring."

    # This display text is for the Gradio UI (Emojis OK)
    status_display = (
        f"{triage}\n\n"
        f"📊 ABCDE Feature Analysis:\n"
        f"- Asymmetry (A): {abcde['asymmetry']}\n"
        f"- Border (B): {abcde['border']}\n"
        f"- Color (C): {abcde['color']}\n"
        f"- Diameter (D): {abcde['diameter']}\n\n"
        f"{'✅' if is_valid else '⚠️'} {validity_msg}\n"
        f"Suggested Action: {action}"
    )

    img_tensor.requires_grad = True
    heatmap = generate_gradcam(model, img_tensor, img, pred_idx)
    
    # Pass status_display; generate_report will clean it
    report = generate_report(diag, conf_str, age, sex, site, status_display, abcde)
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    
    del img_tensor; gc.collect()
    return confidences, status_display, report, heatmap, segmentation_view

# --- 6. UI Construction ---
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Decision Support Prototype")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(type="pil", label="Dermoscopic Input")
            with gr.Row():
                age_in = gr.Number(label="Age", value=45)
                sex_in = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
            site_in = gr.Dropdown(["Face", "Back", "Chest", "Extremity", "Other"], label="Site", value="Back")
            run_btn = gr.Button("🚀 Execute Clinical Audit", variant="primary")
        
        with gr.Column(scale=3):
            with gr.Row():
                output_heatmap = gr.Image(label="Grad-CAM (CNN Focus)")
                output_seg = gr.Image(label="Lesion Segmentation")
            
            output_labels = gr.Label(label="Diagnostic Confidence")
            output_status = gr.Textbox(label="Clinical Indicators & Triage", lines=10)
            output_pdf = gr.File(label="Download Clinical Report")

    run_btn.click(
        fn=predict_clinical, 
        inputs=[input_img, age_in, sex_in, site_in], 
        outputs=[output_labels, output_status, output_pdf, output_heatmap, output_seg]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(), 
        show_error=True
    )