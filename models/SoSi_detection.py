# imports
import torch
import torchvision

# a custom class for the object detector module
class SoSiDetectionModel(torch.nn.Module):
    '''Class for the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
    def __init__(self):
        '''Init the single object, single instance (SoSi) detector utilizing MobileNet V3.'''
        # initialize super
        super().__init__()
        
        # params
        self.num_classes = 2 # 1 + 1 for background
        
        # select backbone according to project definition, with pretrained weights
        pretrained_weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        # take only the features (without softmax layer)
        self.backbone = torchvision.models.mobilenet_v3_large(weights = pretrained_weights).features
        # the output depth of MobilNet Large is fixed at 960
        self.backbone.out_channels = 960 
        self.backbone.out_w = 7
        
        # add the classifier head
        # self.conv = torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        # self.fc = nn.Linear(256 * 7 * 7, num_classes + 4)  # Classes + 4 for AABB
        
        # save the standard transforms - given by default
        self.backbone.transforms = pretrained_weights.transforms
    
    def forward(self, x):
        x = self.backbone(x)
        # x = self.conv(x)
        # x = self.relu(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.fc(x)
        return x