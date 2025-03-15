# models/SoSi_detection.py
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

class SoSiDetectionModel(nn.Module):
    def __init__(
        self,
        freeze_backbone=True,
        final_head_conv_depth=64,
    ):
        super().__init__()
        pretrained_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        
        # Build the entire MobileNet on CPU by default
        mobilenet = mobilenet_v3_large(weights=pretrained_weights)
        self.backbone = mobilenet.features

        # Optionally freeze
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_out_channels = 960
        self.backbone_out_w = 7
        self.final_head_conv_depth = final_head_conv_depth

        # Bbox head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(final_head_conv_depth*((self.backbone_out_w+1)**2), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        # Class head
        self.class_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, final_head_conv_depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_head_conv_depth, 1)
        )

        # Store the transforms
        self.backbone_transforms = pretrained_weights.transforms

    def forward(self, x):
        x = self.backbone(x)
        bbox = self.bbox_head(x).unsqueeze(1)
        class_logits = self.class_head(x)
        return bbox, class_logits
