# imports
import torch
import torchvision

torchvision.models.detection.

# a custom class for the object detector module
class objectDetectionModel(torch.nn.Module):
    # init
    def __init__(self, num_classes):
        '''
        Inputs:
            num_classes - the number of object classes to detect, not incl. background
        '''
        # initialize super
        super(objectDetectionModel, self).__init__()
        
        # params
        self.num_classes = num_classes + 1 # +1 for background
        
        # select backbone according to project definition :)
        self.pretrained_weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.backbone = torchvision.models.mobilenet_v3_large(self.pretrained_weights)
        self.backbone.children
        # 
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv = torch.nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256 * 7 * 7, num_classes + 4)  # Classes + 4 for AABB
        
        # get standard transforms - given by default
        self.pretrained_weights.transforms
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x