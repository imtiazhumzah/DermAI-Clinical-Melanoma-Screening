import gradio as gr
import torch
import os
from PIL import Image
from fpdf import FPDF
from datetime import datetime
from model_utils import get_model, get_transforms, apply_clinical_logic

# --- 1. Setup ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_melanoma_recall_model.pth"
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

model = get_model(num_classes=7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

transform = get_transforms()

# --- 2. Clinical Report Logic ---
def generate_report(image, diagnosis, confidence, age, sex, site, status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="DermAI Clinical Screening Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.line(10, 30, 200, 30)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Metadata:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Age: {age} | Sex: {sex} | Location: {site}", ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diagnostic Results:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"Primary Diagnosis: {diagnosis} ({confidence})", ln=True)
    pdf.multi_cell(0, 10, txt=f"Safety Status: {status}")
    
    report_path = "clinical_report.pdf"
    pdf.output(report_path)
    return report_path

# --- 3. Main Prediction Logic ---
def predict_multimodal(img, age, sex, site):
    if img is None: return None, "Upload required.", None
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES)
    diag = CLASSES[pred_idx]
    conf = f"{probs[pred_idx]:.2%}"
    
    if pred_idx == 4:
        status = "🚨 HIGH RISK: Safety threshold triggered for Melanoma."
    else:
        status = f"✅ Routine Review: Melanoma prob ({mel_prob:.2%}) below threshold."

    # Generate the PDF report immediately
    report = generate_report(img, diag, conf, age, sex, site, status)
    
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, status, report

# --- 4. Professional UI Construction ---
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Multimodal Clinical Screening")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Dermoscopic Image")
            
            with gr.Group():
                gr.Markdown("### Patient Metadata")
                age = gr.Number(label="Patient Age", value=45)
                sex = gr.Radio(["Male", "Female", "Other"], label="Patient Sex", value="Male")
                site = gr.Dropdown(
                    ["Face", "Back", "Chest", "Upper Extremity", "Lower Extremity", "Other"], 
                    label="Anatomical Site", value="Back"
                )
            
            run_btn = gr.Button("Analyze Lesion & Generate Report", variant="primary")
        
        with gr.Column(scale=1):
            output_labels = gr.Label(label="Diagnostic Probabilities")
            output_status = gr.Textbox(label="Clinical Summary")
            output_pdf = gr.File(label="Download Clinical PDF Report")

    run_btn.click(
        fn=predict_multimodal, 
        inputs=[input_img, age, sex, site], 
        outputs=[output_labels, output_status, output_pdf]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())