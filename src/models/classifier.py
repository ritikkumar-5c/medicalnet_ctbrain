"""MedicalNet backbone + classification head for CT-brain (3-class)."""
import torch.nn as nn

from .resnet import build_resnet3d
from .weights import load_pretrained_backbone


class MedicalNetClassifier(nn.Module):
    def __init__(self, depth=34, num_classes=3, in_channels=1, dropout=0.3):
        super().__init__()
        self.backbone = build_resnet3d(depth=depth, in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze


def build_model(depth=34, num_classes=3, in_channels=1, dropout=0.3,
                pretrained=True, pretrained_path=None):
    model = MedicalNetClassifier(
        depth=depth, num_classes=num_classes, in_channels=in_channels, dropout=dropout,
    )
    if pretrained:
        load_pretrained_backbone(model.backbone, depth, pretrained_path=pretrained_path)
    return model
