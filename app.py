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

# Clinical Evidence Mapping (HAM10000 Metadata)
DATASET_EVIDENCE = {
    'AKIEC': "Actinic Keratoses: Precancerous. Often found on sun-damaged skin.",
    'BCC': "Basal Cell Carcinoma: Common skin cancer. Rarely spreads, but locally invasive.",
    'BKL': "Benign Keratosis: Non-cancerous (Seborrheic keratoses or lichen-planus like).",
    'DF': "Dermatofibroma: Benign fibrous tumor. Usually firm and found on legs.",
    'MEL': "MELANOMA: Malignant. High potential for metastasis. Immediate attention required.",
    'NV': "Melanocytic Nevi: Common benign moles. Standard baseline for comparison.",
    'VASC': "Vascular Lesions: Benign angiomas or pyogenic granulomas."
}

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
    validity_msg = "Dermoscopic Likelihood: High" if is_valid else "Image Validity: Low (Possible Blur)"
    return is_valid, reliability, validity_msg

def calculate_abcde(img_cv, mask):
    flipped_h = cv2.flip(mask, 1)
    asymmetry_score = np.sum(cv2.absdiff(mask, flipped_h)) / (np.sum(mask) + 1e-8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness, diameter_px = 0, 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 0: compactness = (perimeter**2) / (4 * np.pi * area)
        _, _, w_box, h_box = cv2.boundingRect(cnt)
        diameter_px = max(w_box, h_box)
    pixels = img_cv[mask > 0]
    color_std = np.std(pixels, axis=0).mean() if pixels.size > 0 else 0
    return {
        "asymmetry": "High" if asymmetry_score > 0.3 else "Low",
        "border": "Irregular" if compactness > 1.6 else "Regular",
        "color": "Polychromatic" if color_std > 35 else "Monochromatic",
        "diameter": f"~{diameter_px * 0.1:.1f} mm" if diameter_px > 0 else "N/A"
    }

# --- 3. Step 3: MC Dropout Uncertainty ---
def predict_with_uncertainty(model, img_tensor, iterations=10):
    model.train() # Enable Dropout
    all_probs = []
    with torch.no_grad():
        for _ in range(iterations):
            logits = model(img_tensor)
            all_probs.append(torch.softmax(logits, dim=1))
    all_probs = torch.cat(all_probs)
    return all_probs.mean(dim=0), all_probs.std(dim=0)

# --- 4. Explainability ---
def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    model.eval()
    target_layer = model.features[7]
    activations, gradients = [], []
    def forward_hook(m, i, o): activations.append(o)
    def backward_hook(m, gi, go): gradients.append(go[0])
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    logits = model(img_tensor)
    model.zero_grad()
    logits[0, target_class_idx].backward()
    grads = gradients[0].detach()
    acts = activations[0].detach()
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = np.maximum(torch.sum(weights * acts, dim=1).squeeze().cpu().numpy(), 0)
    cam = cv2.resize(cv2.GaussianBlur(cam, (11, 11), 0), (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR), 0.7, heatmap, 0.3, 0)
    h1.remove(); h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 5. Clinical Reporting ---
def generate_report(diagnosis, confidence, age, sex, site, status_text, abcde, stability, clinical_note):
    clean_status = status_text.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "").replace("🛡️ ", "").replace("📊 ", "").replace("✅ ", "").replace("⚠️ ", "")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, text="DermAI: Advanced Clinical Audit Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, text=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, 32, 200, 32); pdf.ln(5)

    pdf.set_font("helvetica", 'B', 11)
    pdf.cell(0, 8, text=f"Patient: {age}yo {sex} | Site: {site}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, text=f"Prediction: {diagnosis} ({confidence}) | Stability: {stability}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_font("helvetica", 'B', 10); pdf.cell(0, 8, text="Clinical Definition:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=10); pdf.multi_cell(0, 6, text=clinical_note)
    pdf.ln(3)
    pdf.set_font("helvetica", 'B', 10); pdf.cell(0, 8, text="ABCDE Metrics:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", size=10); pdf.cell(0, 6, text=f"A: {abcde['asymmetry']} | B: {abcde['border']} | C: {abcde['color']} | D: {abcde['diameter']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    report_path = f"/tmp/DermAI_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(report_path)
    return report_path

# --- 6. Main Pipeline ---
def predict_clinical(img, age, sex, site):
    if img is None: return [None] * 5
    gc.collect()
    
    is_valid, reliability, validity_msg = perform_ood_check(img)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    abcde = calculate_abcde(img_cv, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_overlay = img_cv.copy()
    cv2.drawContours(seg_overlay, contours, -1, (0, 255, 0), 2)
    segmentation_view = cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    mean_probs, std_probs = predict_with_uncertainty(model, img_tensor)
    pred_idx, mel_prob = apply_clinical_logic(mean_probs, CLASSES)
    diag = CLASSES[pred_idx]
    
    stability = "High" if float(std_probs[pred_idx]) < 0.05 else "Moderate" if float(std_probs[pred_idx]) < 0.15 else "Low"
    clinical_note = DATASET_EVIDENCE.get(diag, "No specific clinical evidence available.")

    triage = "🔴 HIGH RISK" if (pred_idx == 4 or mel_prob > 0.20) else "🟢 LOW RISK"
    action = "Urgent referral recommended." if "HIGH" in triage else "Routine clinical monitoring."

    status_display = (
        f"{triage}\n\n"
        f"📖 Clinical Note: {clinical_note}\n\n"
        f"📊 ABCDE Analysis:\n- A: {abcde['asymmetry']} | B: {abcde['border']}\n- C: {abcde['color']} | D: {abcde['diameter']}\n\n"
        f"🛡️ Reliability: {validity_msg} | Stability: {stability}\n"
        f"Suggested Action: {action}"
    )

    heatmap = generate_gradcam(model, img_tensor.clone().detach().requires_grad_(True), img, pred_idx)
    report = generate_report(diag, f"{mean_probs[pred_idx]:.2%}", age, sex, site, status_display, abcde, stability, clinical_note)
    confidences = {CLASSES[i]: float(mean_probs[i]) for i in range(len(CLASSES))}
    
    return confidences, status_display, report, heatmap, segmentation_view

# --- 7. UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Advanced Clinical Dashboard")
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
                output_heatmap = gr.Image(label="Grad-CAM Focus Area")
                output_seg = gr.Image(label="Lesion Segmentation")
            output_labels = gr.Label(label="Diagnostic Confidence")
            output_status = gr.Textbox(label="Clinical Summary & Triage", lines=12)
            output_pdf = gr.File(label="Download Clinical Report")

    run_btn.click(fn=predict_clinical, inputs=[input_img, age_in, sex_in, site_in], outputs=[output_labels, output_status, output_pdf, output_heatmap, output_seg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), show_error=True)