import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, swin_t, Swin_T_Weights

def build_model(name="convnext", num_classes=4):
    if name == "convnext":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[2].in_features, num_classes))
    else:
        m = swin_t(weights=Swin_T_Weights.DEFAULT)
        m.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.head.in_features, num_classes))
    return m
