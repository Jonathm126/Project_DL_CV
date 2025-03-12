# imports
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import nms, box_convert

# a custom class for the object detector module
class MoMiDetectionModel(torch.nn.Module):
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
        # num classes + objectness
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
        
        # # objectness?
        # self.obj_head = nn.Sequential(
        #     # 1st conv
        #     nn.Conv2d(960, 256, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout(0.2),
        #     # 2nd conv
        #     nn.Conv2d(256, 128, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Dropout(0.2),
        #     # final conv
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )
    
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
        # Split the classifier output into:
        # - class_pred: all channels except the last (shape: [B, 49, num_classes])
        # - objectness_pred: last channel (shape: [B, 49, 1])
        class_pred = cls_pred[..., :-1]
        objectness_pred = cls_pred[..., -1:]
        
        return bbox_pred, class_pred, objectness_pred
    

    def inference(self, images, obj_threshold=0.5, nms_threshold=0.4):
        """
        Performs inference on images, filtering predictions by objectness threshold and applying non-maximum suppression (NMS).
        Args:
            images (Tensor): Input images tensor [B, 3, H, W]
            obj_threshold (float): Objectness threshold (e.g., 0.5)
            nms_threshold (float): NMS IoU threshold (e.g., 0.4)
        Returns:
            detections: List of detections for each image. Each detection might be a tuple of (bbox, class, score) after NMS.
        """
        # self.eval()
        pred_boxes, pred_labels_logits, pred_obj_logits = self.forward(images)
        
        # objectness
        obj_probs = torch.sigmoid(pred_obj_logits).squeeze(-1)  # [B, 49, 1]
        obj_scores = obj_probs.squeeze(-1)  # [B, 49]
        
        # classes
        class_probs = torch.softmax(pred_labels_logits, dim=-1)  # [B, 49, num_classes]
        class_scores, class_preds = class_probs.max(dim=-1)         # [B, 49]
        
        #  combined score as the product of class confidence and objectness
        bbox_scores = class_scores * obj_scores  # shape: [B, 49]
        
        all_boxes = []
        all_labels = []
        all_confs = []
        
        B = images.size(0)
        for b in range(B):
            # Filter grid cells by objectness threshold.
            mask = bbox_scores[b] > obj_threshold
            boxes = pred_boxes[b][mask]  # [N, 4]
            labels = class_preds[b][mask]  # [N]
            confs = bbox_scores[b][mask]  # [N]
            
            # Apply NMS using torchvision's built-in function.
            if boxes.numel() > 0:
                # Convert boxes from xywh to xyxy for NMS.
                boxes_xyxy = box_convert(boxes, 'xywh', 'xyxy')
                keep = nms(boxes_xyxy, confs, nms_threshold)
                boxes = boxes[keep]
                labels = labels[keep]
                confs = confs[keep]
            else:
                boxes = torch.empty((0, 4), device=images.device)
                labels = torch.empty((0,), dtype=torch.long, device=images.device)
                confs = torch.empty((0,), device=images.device)
            
            all_boxes.append(boxes)
            all_labels.append(labels)
            all_confs.append(confs)
        
        return torch.stack(all_boxes), torch.stack(all_labels), torch.stack(all_confs)
