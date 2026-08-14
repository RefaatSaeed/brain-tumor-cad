import timm
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, swin_t, Swin_T_Weights


def build_model(model_name: str = "convnext_tiny", num_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
    """
    Factory function to instantiate state-of-the-art vision backbones.
    
    Supported backbones:
      - 'convnext_tiny': Modern pure convolutional architecture
      - 'swin_tiny'    : Hierarchical Shifted-Window Vision Transformer
    """
    model_name = model_name.lower()

    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        
        # Replace classification head
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "swin_tiny":
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = swin_t(weights=weights)
        
        # Replace classification head
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    else:
        # Fallback to timm for additional backbones (e.g. efficientnetv2)
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)

    return model