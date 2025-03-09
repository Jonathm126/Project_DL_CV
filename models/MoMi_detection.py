# imports
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# a custom class for the object detector module
class SoSiDetectionModel(torch.nn.Module):
    '''Class for the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
    def __init__(self, num_classes, freeze_backbone = True, final_head_conv_depth = 64):
        '''Init the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
        # initialize super
        super().__init__()
        self.num_classes = num_classes
        
        # select backbone according to project definition, with pretrained weights and transforms
        pretrained_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.backbone = mobilenet_v3_large(weights = pretrained_weights).features
        self.backbone_transforms = pretrained_weights.transforms
        
        # freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # the output depth of MobilNet Large is fixed at 960
        self.backbone_out_channels = 960 
        self.backbone_out_w = 7
        self.final_head_conv_depth = final_head_conv_depth
        
        # predict 4 bbox coordinates
        self.bbox_head = nn.Sequential(
            # 1st conv
            nn.Conv2d(960, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            # 2nd conv
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            # final conv
            nn.Conv2d(128, 4, kernel_size=1),
            nn.Sigmoid()  # Outputs normalized between 0 and 
        )
        
        # classifier
        self.cls_head = nn.Sequential(
            # 1st conv
            nn.Conv2d(960, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            # 2nd conv
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            # final conv
            nn.Conv2d(128, num_classes + 1, kernel_size=1)
        )
        
        # objectness?
        self.obj_head = nn.Sequential(
            # 1st conv
            nn.Conv2d(960, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            # 2nd conv
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            # final conv
            nn.Conv2d(128, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract features: shape [B, 960, 7, 7]
        features = self.backbone(x)
        # Bounding box regression: output shape [B, 4, 7, 7]
        bbox_pred = self.bbox_head(features)
        # Reshape to [B, 49, 4] (since 7x7=49 spatial locations)
        bbox_pred = bbox_pred.view(x.size(0), 4, -1).permute(0, 2, 1)
        # Classification prediction: output shape [B, num_classes, 7, 7]
        cls_pred = self.cls_head(features)
        # Reshape to [B, 49, num_classes]
        cls_pred = cls_pred.view(x.size(0), cls_pred.size(1), -1).permute(0, 2, 1)    
        # Objectness prediction: output shape [B, 1, 7, 7]
        obj_pred = self.objectness_head(features)
        # Reshape to [B, 49, 1]
        obj_pred = obj_pred.view(x.size(0), 1, -1).permute(0, 2, 1)
        
        return bbox_pred, cls_pred, obj_pred
