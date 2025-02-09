# imports
import torch
import torch.nn as nn
import torchvision

# a custom class for the object detector module
class SoSiDetectionModel(torch.nn.Module):
    '''Class for the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
    def __init__(self, shared_head_conv_depth = 64):
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
        self.shared_head_conv_depth = shared_head_conv_depth
        
        # the final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, self.shared_head_conv_depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),  
        )
        # predict 4 bbox coordinates
        self.bbox_head = nn.Linear(self.shared_head_conv_depth, 4)  
        # predict object presence
        self.class_head = nn.Linear(self.shared_head_conv_depth, 1)  
        
        # save the standard transforms - given by default
        self.backbone_transforms = pretrained_weights.transforms
    
    def forward(self, x):
        # find feature vector
        x = self.backbone(x)
        # one final conv to shared_head_conv_depth
        x = self.final_conv(x)
        # use the bounding box head
        bbox = self.bbox_head(x)
        # use the classifying head
        class_logits = self.class_head(x)

        return bbox, class_logits
