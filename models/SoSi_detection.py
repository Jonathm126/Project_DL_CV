# imports
import torch
import torch.nn as nn
import torchvision

# a custom class for the object detector module
class SoSiDetectionModel(torch.nn.Module):
    '''Class for the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
    def __init__(self):
        '''Init the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
        # initialize super
        super().__init__()
        
        # select backbone according to project definition, with pretrained weights
        pretrained_weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        
        # take only the features (without softmax layer)
        self.backbone = torchvision.models.mobilenet_v3_large(weights = pretrained_weights).features
        
        # the output depth of MobilNet Large is fixed at 960
        self.backbone_out_channels = 960 
        self.backbone_out_w = 7
        
        # bounding box head: outputs 4 values (x_min, y_min, x_max, y_max)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Pool to a 1x1 spatial resolution
            nn.Flatten(),  # Flatten to convert to 1D for fully connected layer
            nn.Linear(256, 4)  # Output 4 bounding box values
        )
        
        # classifier head: outputs 1 value for the class prediction (0 or 1)
        self.class_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Pool to a 1x1 spatial resolution
            nn.Flatten(),  # Flatten to convert to 1D for fully connected layer
            nn.Linear(256, 1),  # Output 1 for binary classification (object presence)
        )
        
        # save the standard transforms - given by default
        self.backbone_transforms = pretrained_weights.transforms
    
    def forward(self, x):
        # find feature vector
        x = self.backbone(x)
        # use the bounding box head
        bbox = self.bbox_head(x)
        # use the classifying head
        class_logits = self.class_head(x)

        return bbox, class_logits
