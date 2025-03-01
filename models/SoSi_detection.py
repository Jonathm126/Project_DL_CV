# imports
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import clip_boxes_to_image

# a custom class for the object detector module
class SoSiDetectionModel(torch.nn.Module):
    '''Class for the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
    def __init__(self, freeze_backbone = True, final_head_conv_depth = 64):
        '''Init the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
        # initialize super
        super().__init__()
        
        # select backbone according to project definition, with pretrained weights
        pretrained_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        
        # take only the features (without softmax layer)
        self.backbone = mobilenet_v3_large(weights = pretrained_weights).features
        
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
            nn.Conv2d(self.backbone_out_channels, 128, kernel_size=2, stride=1, padding=1), # output 7X7
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            # nn.AdaptiveAvgPool2d(1),
            # 2nd conv
            # nn.Conv2d(128, self.final_head_conv_depth, kernel_size=3, stride=1, padding=1), # output 3X3
            # nn.ReLU(),
            # nn.BatchNorm2d(self.final_head_conv_depth),
            # nn.Dropout(0.3),
            
            # start fc layer
            nn.Flatten(),  
            
            # 1st fc
            nn.Linear(self.final_head_conv_depth * ((self.backbone_out_w) ** 2), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 2nd fc
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # final fc
            nn.Linear(64, 4),
            
            # add sigmoid since the output is normalized
            nn.Sigmoid()
        )
        
        # self.bbox_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.backbone_out_channels * (self.backbone_out_w ** 2), 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 4),
        #     nn.Sigmoid()
        # )
        
        # predict object presence
        self.class_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, self.final_head_conv_depth, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),  
            nn.Linear(self.final_head_conv_depth, 1)
        )
        
        # save the standard transforms - given by default
        self.backbone_transforms = pretrained_weights.transforms
    
    def forward(self, x):
        # find feature vector
        x = self.backbone(x)
        # use the bounding box head
        bbox = self.bbox_head(x)
        # clip the bbox to 0,1 - bbox is normalized
        # bbox = clip_boxes_to_image(bbox, (1,1))
        # shape the output to [B,N,4] to match the multi instance case
        bbox = bbox.unsqueeze(1)
        # get the classifier
        class_logits = self.class_head(x)
        # shape the output to [B,N,1] to match the multi instance case
        class_logits = class_logits
        
        return bbox, class_logits
