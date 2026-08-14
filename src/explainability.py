import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model, model_name: str):
    """
    Selects the optimal final feature extraction layer for Grad-CAM.
    """
    model_name = model_name.lower()
    if "convnext" in model_name:
        # Last convolutional block of ConvNeXt
        return [model.features[-1][-1].block]
    elif "swin" in model_name:
        # Last Norm layer before pooling in Swin Transformer
        return [model.norm]
    else:
        raise ValueError(f"Target layer not defined for: {model_name}")


def generate_gradcam_overlay(model, model_name: str, input_tensor: torch.Tensor, original_rgb: np.ndarray, target_class_idx: int = None):
    """
    Computes Grad-CAM activation map and blends it onto the MRI slice.
    
    Args:
        model: Trained PyTorch neural network.
        model_name: 'convnext_tiny' or 'swin_tiny'
        input_tensor: Normalized (1, 3, 224, 224) torch tensor.
        original_rgb: (224, 224, 3) float32 numpy image in range [0, 1].
        target_class_idx: Integer class to compute gradients for (default: argmax).
    """
    model.eval()
    target_layers = get_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Normalize original RGB slice to [0, 1]
        if original_rgb.max() > 1.0:
            rgb_float = original_rgb.astype(np.float32) / 255.0
        else:
            rgb_float = original_rgb.astype(np.float32)

        # Blend CAM heatmap with original MRI
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam