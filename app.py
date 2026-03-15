import gradio as gr
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import gc
import re
from PIL import Image
from fpdf import FPDF, XPos, YPos
from datetime import datetime

# Import your architecture and logic from local files
from model_utils import get_model, get_transforms, apply_clinical_logic

# --- 1. Setup & Configuration ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_melanoma_recall_model.pth"
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

DATASET_EVIDENCE = {
    'AKIEC': "Actinic Keratoses: Precancerous lesion. Common in sun-damaged skin.",
    'BCC': "Basal Cell Carcinoma: Common skin cancer. Locally invasive.",
    'BKL': "Benign Keratosis: Non-cancerous seborrheic keratosis.",
    'DF': "Dermatofibroma: Benign fibrous nodule.",
    'MEL': "MELANOMA: Malignant. Requires urgent surgical evaluation.",
    'NV': "Melanocytic Nevi: Benign common mole.",
    'VASC': "Vascular Lesion: Benign angioma."
}

model = get_model(num_classes=7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = get_transforms()

# --- 2. Advanced Analysis Logic ---
def perform_ood_check(image):
    img_np = np.array(image.convert("L"))
    sharpness = cv2.Laplacian(img_np, cv2.CV_64F).var()
    is_valid = sharpness > 50 
    return is_valid, ("High" if is_valid else "Reduced")

def calculate_abcde(img_cv, mask):
    # A - Asymmetry
    flipped_h = cv2.flip(mask, 1)
    asymmetry = np.sum(cv2.absdiff(mask, flipped_h)) / (np.sum(mask) + 1e-8)
    # B - Border & D - Diameter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compactness, diam_px = 0, 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area > 0: compactness = (peri**2) / (4 * np.pi * area)
        _, _, w, h = cv2.boundingRect(cnt)
        diam_px = max(w, h)
    # C - Color
    pixels = img_cv[mask > 0]
    color_std = np.std(pixels, axis=0).mean() if pixels.size > 0 else 0
    return {
        "A": "High" if asymmetry > 0.3 else "Low",
        "B": "Irregular" if compactness > 1.6 else "Regular",
        "C": "Polychromatic" if color_std > 35 else "Monochromatic",
        "D": f"~{diam_px * 0.1:.1f} mm"
    }

def predict_with_uncertainty(model, img_tensor, iterations=10):
    model.train() # Enable Dropout
    all_probs = []
    with torch.no_grad():
        for _ in range(iterations):
            all_probs.append(torch.softmax(model(img_tensor), dim=1))
    all_probs = torch.cat(all_probs)
    return all_probs.mean(dim=0), all_probs.std(dim=0)

def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    model.eval()
    target_layer = model.features[7]
    activations, gradients = [], []
    def f_hook(m, i, o): activations.append(o)
    def b_hook(m, gi, go): gradients.append(go[0])
    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)
    
    logits = model(img_tensor)
    model.zero_grad()
    logits[0, target_class_idx].backward()
    
    grads, acts = gradients[0].detach(), activations[0].detach()
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = np.maximum(torch.sum(weights * acts, dim=1).squeeze().cpu().numpy(), 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR), 0.7, heatmap, 0.3, 0)
    h1.remove(); h2.remove()
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- 3. Clinical Reporting (Robust Emoji Fix) ---
def generate_report(diag, conf, age, sex, site, status, abcde, stability):
    # Strip emojis for PDF compatibility
    clean_s = re.sub(r'[^\x00-\x7F]+', '', status)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, text="DermAI Clinical Audit Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, text=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, 32, 200, 32); pdf.ln(5)
    pdf.set_font("helvetica", 'B', 11)
    pdf.cell(0, 8, text=f"Patient: {age}yo {sex} | Site: {site}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, text=f"Finding: {diag} ({conf}) | Stability: {stability}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    pdf.multi_cell(0, 7, text=f"Clinical Summary:\n{clean_s}")
    report_path = f"/tmp/DermAI_Report.pdf"
    pdf.output(report_path)
    return report_path

# --- 4. Main Controller ---
def clinical_workflow(img, age, sex, site):
    if img is None: return [None]*6
    gc.collect()
    
    # 1. Vision Logic
    is_valid, reliability = perform_ood_check(img)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    abcde = calculate_abcde(img_cv, mask)
    
    # 2. ML Logic
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    mean_probs, std_probs = predict_with_uncertainty(model, img_tensor)
    idx, mel_prob = apply_clinical_logic(mean_probs, CLASSES)
    diag = CLASSES[idx]
    
    # 3. Descriptive Mapping
    stability_val = float(std_probs[idx])
    stability_text = "High" if stability_val < 0.08 else "Moderate" if stability_val < 0.15 else "Low"
    conf_text = "High" if float(mean_probs[idx]) > 0.85 else "Moderate" if float(mean_probs[idx]) > 0.60 else "Low"
    
    a_desc = "High asymmetry" if abcde['A'] == "High" else "Low asymmetry"
    b_desc = "Irregular border" if abcde['B'] == "Irregular" else "Regular border"
    c_desc = "High color variation" if abcde['C'] == "Polychromatic" else "Low color variation"
    d_desc = f"Estimated diameter {abcde['D']}"

    if idx == 4 or mel_prob > 0.20:
        triage_header = "🔴 HIGH RISK LESION"
        assessment = "High probability of melanoma-like characteristics detected."
        guidance = "Dermatology consultation recommended. Further clinical evaluation, including dermoscopy or biopsy, may be required."
    else:
        triage_header = "🟢 LOW RISK LESION"
        assessment = "Lesion exhibits characteristics typical of benign findings."
        guidance = "Routine clinical monitoring suggested. Review if changes in size, shape, or color occur."

    status_ui = (
        f"{triage_header}\n\n"
        f"AI Assessment:\n{assessment}\n\n"
        f"Clinical Guidance:\n{guidance}\n\n"
        f"📊 ABCDE Indicators\n"
        f"A: {a_desc}\n"
        f"B: {b_desc}\n"
        f"C: {c_desc}\n"
        f"D: {d_desc}\n\n"
        f"🛡️ Model Reliability\n"
        f"Prediction Confidence: {conf_text}\n"
        f"Model Stability: {stability_text}"
    )

    # 4. Visuals & Report
    heatmap = generate_gradcam(model, img_tensor.clone().detach().requires_grad_(True), img, idx)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_view = cv2.drawContours(img_cv.copy(), contours, -1, (0,255,0), 2)
    report = generate_report(diag, f"{mean_probs[idx]:.1%}", age, sex, site, status_ui, abcde, stability_text)
    
    confidences = {CLASSES[i]: float(mean_probs[i]) for i in range(len(CLASSES))}
    return confidences, status_ui, report, heatmap, cv2.cvtColor(seg_view, cv2.COLOR_BGR2RGB), stability_text

# --- 5. UI Layout (4-Page Design) ---
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Enterprise Dermatology Support")
    
    with gr.Tabs():
        # PAGE 1: PRIMARY SCREENING
        with gr.Tab("📋 Patient Audit"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="Dermoscopic Upload")
                    with gr.Row():
                        age_in = gr.Number(label="Age", value=45)
                        sex_in = gr.Radio(["Male", "Female"], label="Sex")
                    site_in = gr.Dropdown(["Face", "Back", "Chest", "Extremity"], label="Site")
                    run_btn = gr.Button("🚀 Execute Audit", variant="primary")
                with gr.Column(scale=1):
                    out_label = gr.Label(label="Diagnostic Probabilities")
                    out_status = gr.Textbox(label="Triage Summary", lines=15)
                    out_pdf = gr.File(label="Clinical PDF Report")

        # PAGE 2: CLINICAL EXPLAINABILITY
        with gr.Tab("🔬 XAI & Metrics"):
            gr.Markdown("### Computer Vision Feature Extraction")
            with gr.Row():
                out_heat = gr.Image(label="Grad-CAM Focus Area")
                out_seg = gr.Image(label="Lesion Segmentation Boundary")
            gr.Markdown("#### Objective Geometric Findings")
            gr.Markdown("Visual evidence layers used by the model for classification and triage determination.")

        # PAGE 3: MODEL CARD & INTEGRITY
        with gr.Tab("🛡️ Model Transparency"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 📜 Model Specification
                    - **Architecture:** ConvNeXt-Tiny (Fine-tuned)
                    - **Training Objective:** Clinical Triage Support
                    - **Uncertainty Method:** Monte Carlo Dropout (N=10)
                    - **Clinical Logic:** Recall-prioritized for Melanoma detection.
                    """)
                with gr.Column():
                    gr.Markdown("### Uncertainty Analysis")
                    out_stability = gr.Textbox(label="Prediction Stability Score")

        # PAGE 4: DATASET ARCHIVE
        with gr.Tab("📊 Dataset Insights"):
            gr.Markdown("### Training Foundation: HAM10000")
            gr.Markdown("Benchmarked against 10,015 expert-labeled dermatoscopic samples.")
            gr.DataFrame(
                headers=["Code", "Condition", "Frequency"],
                value=[["MEL", "Melanoma", "11%"], ["NV", "Nevi", "67%"], ["BCC", "Basal Cell", "5%"]]
            )

    run_btn.click(
        fn=clinical_workflow,
        inputs=[input_img, age_in, sex_in, site_in],
        outputs=[out_label, out_status, out_pdf, out_heat, out_seg, out_stability]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(primary_hue="blue"), 
        show_error=True
    )