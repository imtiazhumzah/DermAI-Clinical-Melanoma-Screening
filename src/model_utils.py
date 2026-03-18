import torch
import torch.nn as nn
from torchvision import transforms, models

def get_model(num_classes=7):
    """
    Initializes the ConvNeXt-Tiny architecture.
    Ensures the final linear layer matches the 7-class diagnostic task.
    """
    model = models.convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def get_transforms():
    """
    Returns the standard medical imaging preprocessing pipeline.
    Uses CenterCrop to remove peripheral lens artifacts as identified in the audit.
    """
    return transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def apply_clinical_logic(probs, classes, threshold=0.20):
    """
    Implements the 0.20 Safety Threshold for Melanoma (MEL).
    
    Args:
        probs (torch.Tensor): Softmax probabilities from the model.
        classes (list): List of diagnostic class names.
        threshold (float): The recall-optimized safety threshold.
        
    Returns:
        int: The final predicted class index.
        float: The probability of the Melanoma class.
    """
    # Melanoma is index 4 in our class list
    mel_prob = float(probs[4])
    
    if mel_prob >= threshold:
        # Override to Melanoma if it passes the safety threshold
        return 4, mel_prob
    else:
        # Otherwise, take the standard argmax
        return torch.argmax(probs).item(), mel_prob