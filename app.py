import gradio as gr
import torch
import os
from PIL import Image

# Import the professional modules we built
from model_utils import get_model, get_transforms, apply_clinical_logic

# --- 1. Configuration & Loading ---
DEVICE = torch.device("cpu")
MODEL_PATH = "best_melanoma_recall_model.pth"
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# Initialize Model
model = get_model(num_classes=7)

# Load weights safely
if os.path.exists(MODEL_PATH):
    # map_location ensures it loads on CPU even if trained on GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model weights loaded from {MODEL_PATH}")
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found. Check your file uploads.")

# Initialize Preprocessing
transform = get_transforms()

# --- 2. Prediction Function ---
def predict(img):
    if img is None:
        return None, "Please upload a dermoscopic image to begin."
    
    # Preprocess and prepare for model
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    # Apply our specialized 0.20 safety logic from model_utils
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES, threshold=0.20)
    
    # Create clinical feedback message
    if pred_idx == 4: # If the model (or override) flags Melanoma
        status = (
            f"🚨 CLINICAL ALERT: High Sensitivity Flag\n"
            f"Melanoma Prob: {mel_prob:.2%}\n"
            f"Action: Urgent Specialist Review Recommended."
        )
    else:
        status = (
            f"✅ Routine Screening\n"
            f"Melanoma Prob: {mel_prob:.2%}\n"
            f"Primary Prediction: {CLASSES[pred_idx]}"
        )

    # Format for Gradio Label component
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, status

# --- 3. Gradio Interface Construction ---
with gr.Blocks as demo:
    gr.Markdown("# 🩺 DermAI: Clinical Malignancy Screening")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Dermoscopic Image Input")
            run_btn = gr.Button("Conduct Clinical Audit", variant="primary")
            
            with gr.Accordion("📋 Model Audit Specs", open=False):
                gr.Markdown("""
                - **Recall (MEL):** 94.64%
                - **ROC-AUC:** 0.9564
                - **Safety Policy:** 20% Sensitivity Override enabled.
                """)
        
        with gr.Column(scale=1):
            output_labels = gr.Label(label="Diagnostic Confidence", num_top_classes=3)
            output_status = gr.Textbox(label="Clinical Recommendation", lines=4)

    run_btn.click(
        fn=predict, 
        inputs=input_img, 
        outputs=[output_labels, output_status]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Note:** This tool is for research purposes. All high-probability flags should be confirmed via biopsy by a certified professional."
    )

# --- 4. Docker Launch Settings ---
if __name__ == "__main__":
    # server_name="0.0.0.0" and server_port=7860 are REQUIRED for Hugging Face Docker Spaces
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_api=False
    )