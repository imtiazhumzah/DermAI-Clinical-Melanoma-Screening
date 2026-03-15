import cv2
import numpy as np
import torch.nn.functional as F

# --- New Function: Grad-CAM Generator ---
def generate_gradcam(model, img_tensor, original_image, target_class_idx):
    """
    Generates a Grad-CAM heatmap overlay.
    """
    # 1. Capture gradients and feature maps from the final conv layer
    # For ConvNeXt/ResNet, this is usually the last block of the feature extractor
    target_layer = dict([*model.named_modules()])['features'][-1] 
    
    activations = []
    gradients = []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # 2. Forward pass
    logits = model(img_tensor)
    
    # 3. Backward pass for the target class
    model.zero_grad()
    score = logits[0, target_class_idx]
    score.backward()

    # 4. Compute Grad-CAM
    grads = gradients[0].detach()
    acts = activations[0].detach()
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze().cpu().numpy()
    
    # 5. Post-process Heatmap
    cam = np.maximum(cam, 0) # ReLU
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # 6. Overlay on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(original_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    # Cleanup hooks
    h1.remove()
    h2.remove()
    
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- Update predict_multimodal to return the heatmap ---
def predict_multimodal(img, age, sex, site):
    if img is None: return None, "Upload required.", None, None
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True # Required for Grad-CAM
    
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    
    pred_idx, mel_prob = apply_clinical_logic(probs, CLASSES)
    diag = CLASSES[pred_idx]
    
    # Generate Grad-CAM Heatmap
    heatmap_img = generate_gradcam(model, img_tensor, img, pred_idx)
    
    # (Rest of your reporting logic remains the same)
    status = f"Diagnosis: {diag}"
    report = generate_report(img, diag, f"{probs[pred_idx]:.2%}", age, sex, site, status)
    
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return confidences, status, report, heatmap_img

with gr.Blocks() as demo:
    gr.Markdown("# 🩺 DermAI: Multimodal Clinical Screening & Explainability")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Dermoscopic Input")
            # ... (Age/Sex/Site inputs) ...
            run_btn = gr.Button("Analyze Lesion", variant="primary")
        
        with gr.Column():
            output_labels = gr.Label(label="Diagnostic Probabilities")
            # NEW: Grad-CAM Display
            output_heatmap = gr.Image(label="Explainability: Grad-CAM Heatmap Overlay")
            output_pdf = gr.File(label="Download Clinical PDF Report")

    run_btn.click(
        fn=predict_multimodal, 
        inputs=[input_img, age, sex, site], 
        outputs=[output_labels, output_status, output_pdf, output_heatmap] # Added output_heatmap
    )