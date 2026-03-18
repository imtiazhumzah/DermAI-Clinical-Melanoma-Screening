---
title: DermAI-Clinical-Melanoma-Screening
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
short_description: Melanoma screening via ConvNeXt & HAM10000 dataset
models:
- imtiazhumzah/DermAI-Clinical-Screen
datasets:
- kuchikihater/HAM10000
tags:
- medical-imaging
- dermatology
- computer-vision
- explainable-ai
- mc-dropout
---

# 🩺 DermAI: Advanced Clinical Melanoma Screening
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/imtiazhumzah/DermAI-Clinical-Melanoma-Screening)

**DermAI** is a state-of-the-art decision support tool designed for clinical-grade dermatoscopic analysis. Moving beyond simple classification, DermAI implements an audit-ready pipeline that provides clinicians with diagnostic probabilities, uncertainty estimations, and visual evidence.

## 🚀 Key Innovations
* **Precision Architecture:** Powered by a fine-tuned **ConvNeXt-Tiny** backbone, specifically optimized for high-sensitivity melanoma detection.
* **Clinical Triage Suite:** A 4-page diagnostic dashboard separating patient intake, deep-dive XAI analytics, and model integrity reports.
* **Uncertainty Quantificaton:** Implements **Monte Carlo (MC) Dropout** ($N=10$) to generate a "Model Stability Score," alerting physicians when the AI is unsure.
* **Visual Audit (XAI):** Real-time **Grad-CAM** heatmaps and automated lesion segmentation to visualize the neural network's focus areas.

## 🔬 Diagnostic Features (ABCDE Analysis)
DermAI automates the standard clinical ABCDE criteria through computer vision:
- **(A) Asymmetry:** Quantitative symmetry analysis across orthogonal axes.
- **(B) Border:** Evaluation of compactness and boundary irregularity.
- **(C) Color:** Polychromatic variance analysis in RGB and HSV spaces.
- **(D) Diameter:** Estimated lesion sizing based on pixel-to-mm scaling.

## 📊 Dataset & Training
The model was trained using the **HAM10000** ("Human Against Machine") dataset, a benchmark in medical AI for dermatology. 

- **Total Images:** 10,015 dermatoscopic samples.
- **Source:** Multi-source collection from the Medical University of Vienna and the University of Queensland.
- **Verification:** Over 50% of cases are histopathology-confirmed (the gold standard in medical diagnostics).
- **Access:** [View Dataset on Hugging Face](https://huggingface.co/datasets/kuchikihater/HAM10000)

- **Preprocessing:** Focal Loss for class imbalance, Test-Time Augmentation (TTA), and ImageNet-standard normalization.
- **Performance Goal:** Prioritized **Recall for Melanoma (MEL)** to minimize false negatives in a screening context.

## 🛡️ Medical Disclaimer
*This tool is intended for research and educational purposes to assist clinical decision-making. It is not a substitute for a biopsy or professional dermatological diagnosis. Always consult a board-certified dermatologist for skin concerns.*

## 🛠️ Tech Stack
- **Inference Engine:** PyTorch & Torchvision
- **Frontend:** Gradio (4-page Clinical Dashboard)
- **Deployment:** Docker on Hugging Face Spaces
- **Reporting:** FPDF2 for automated clinical PDF generation

---
**Developed by:** [Imtiaz Humzah](https://huggingface.co/imtiazhumzah)