import math
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.ops import nms

class MoMiDetectionModel(nn.Module):
    """
    A single-class multi-instance object detector using a MobileNet V3 backbone
    and multiple anchors per grid cell (similar to YOLO/SSD).
    """
    def __init__(
        self, 
        num_classes=1, 
        freeze_backbone=True, 
        final_head_conv_depth=64,
        anchors=None
    ):
        """
        Args:
          num_classes (int): # of foreground classes (excluding background).
                             e.g. 1 => 'cat' (plus background => 2 classes total).
          freeze_backbone (bool): Whether to freeze the MobileNet V3 layers.
          final_head_conv_depth (int): Channel dimension for final conv layers (not heavily used).
          anchors (list): e.g. [(0.05,1.0), (0.1,1.0), (0.2,1.0)] => anchor scales & aspect ratios.
        """
        super().__init__()
        
        # We add +1 for background => total classification channels = num_classes+1
        self.num_classes = num_classes
        self.num_classes_plus_bg = num_classes + 1
        
        # Load pretrained MobileNet V3 Large from torchvision
        pretrained_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        # .features => the feature-extraction part
        self.backbone = mobilenet_v3_large(weights=pretrained_weights).features
        # transformations for normalizing, resizing, etc.
        self.backbone_transforms = pretrained_weights.transforms

        # Optionally freeze backbone => so only heads are trainable
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # typically => [B, 960, 7,7] for input ~224x224
        self.backbone_out_channels = 960
        # we assume final spatial size = 7 x 7
        self.feature_map_size = 7
        self.final_head_conv_depth = final_head_conv_depth
        
        # If no anchors are provided, default to 3 squares
        if anchors is None:
            anchors = [
                (0.05, 1.0),  # small
                (0.1,  1.0),  # medium
                (0.2,  1.0),  # large
            ]
        self.anchors = anchors
        self.num_anchors = len(anchors)
        
        # BBox head => predicts 4 offsets per anchor => total channels = 4 * num_anchors
        self.bbox_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            
            nn.Conv2d(128, 4 * self.num_anchors, kernel_size=1)
        )
        
        # Classification head => (num_classes_plus_bg)*num_anchors => e.g. 2*k if num_classes=1
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            
            nn.Conv2d(128, self.num_anchors * self.num_classes_plus_bg, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass:
          1) run MobileNet V3 => 'feats' shape [B, 960, 7,7]
          2) pass feats to bbox_head => offsets => [B, 4*k, 7,7] => reshape to [B, 49*k, 4]
          3) pass feats to cls_head  => classification => [B, (num_classes+1)*k, 7,7] => reshape to [B,49*k,(num_classes+1)]
        
        Args:
          x => [B, C, H, W]

        Returns:
          bbox_out => [B, (49*k), 4]
          cls_out  => [B, (49*k), (num_classes+1)]
        """
        B = x.size(0)
        
        # 1) backbone feature extraction
        feats = self.backbone(x)  # e.g. [B, 960, 7,7]
        
        # 2) bounding-box offsets
        bbox_out = self.bbox_head(feats)
        # => shape [B, 4*k, 7,7]
        # reorder to => [B, 49*k, 4]
        bbox_out = bbox_out.view(
            B,
            4*self.num_anchors,
            self.feature_map_size**2
        )  # => [B, 4*k, 49]
        bbox_out = bbox_out.permute(0, 2, 1).contiguous()   # => [B,49,4*k]
        bbox_out = bbox_out.view(
            B,
            self.feature_map_size*self.feature_map_size*self.num_anchors,
            4
        )
        
        # 3) classification logits => [B, (num_classes_plus_bg)*k, 7,7]
        cls_out = self.cls_head(feats)
        # reshape => [B, 49*k, (num_classes+1)]
        cls_out = cls_out.view(
            B,
            self.num_classes_plus_bg*self.num_anchors,
            self.feature_map_size**2
        )
        cls_out = cls_out.permute(0, 2, 1).contiguous()  # => [B,49,2*k] if single class
        cls_out = cls_out.view(
            B,
            self.feature_map_size*self.feature_map_size*self.num_anchors,
            self.num_classes_plus_bg
        )
        
        return bbox_out, cls_out
    
    def inference(self, images, obj_threshold=0.5, nms_threshold=0.4):
        """
        Full detection inference:
          - forward pass => predicted offsets + class logits
          - decode offsets => bounding boxes
          - cat probability => softmax (index=1)
          - filter by obj_threshold
          - run NMS => final detections
        
        Args:
          images => [B, C, H, W]
          obj_threshold => min confidence for a box to be kept
          nms_threshold => IoU threshold for Non-Max Suppression
        
        Returns:
          all_boxes, all_labels, all_scores: each is a list of length B
            e.g. all_boxes[b] => shape [N,4]
                 all_labels[b]=> shape [N], 0 => cat
                 all_scores[b]=> shape [N]
        """
        self.eval()
        with torch.no_grad():
            # 1) get predictions
            bbox_out, cls_out = self.forward(images)
            B, nAnchors, _ = bbox_out.shape
            
            # 2) cat probability => softmax => [B,nAnchors,2]
            probs = torch.softmax(cls_out, dim=-1)
            cat_prob = probs[..., 1]  # index=1 => cat

            # DEBUG: print the maximum cat probability for the entire batch
            max_cat_prob_val = cat_prob.max().item()
            print(f"[DEBUG inference] max cat_prob across batch = {max_cat_prob_val:.4f}")
            
            # 3) build base anchors => same as in training
            base_anchors = []
            for row in range(self.feature_map_size):
                for col in range(self.feature_map_size):
                    cx = (col+0.5)/self.feature_map_size
                    cy = (row+0.5)/self.feature_map_size
                    for (scale, aspect) in self.anchors:
                        w = scale * math.sqrt(aspect)
                        h = scale / math.sqrt(aspect)
                        base_anchors.append([cx, cy, w, h])
            
            base_anchors = torch.tensor(
                base_anchors, dtype=torch.float32, device=images.device
            )
            
            # 4) decode offsets => anchor + predicted offsets => bounding boxes
            tx = bbox_out[..., 0]
            ty = bbox_out[..., 1]
            tw = bbox_out[..., 2]
            th = bbox_out[..., 3]

            anc_cx = base_anchors[:, 0].unsqueeze(0).expand(B, -1)
            anc_cy = base_anchors[:, 1].unsqueeze(0).expand(B, -1)
            anc_w  = base_anchors[:, 2].unsqueeze(0).expand(B, -1)
            anc_h  = base_anchors[:, 3].unsqueeze(0).expand(B, -1)
            
            pred_cx = anc_cx + tx * anc_w
            pred_cy = anc_cy + ty * anc_h
            pred_w  = anc_w * torch.exp(tw)
            pred_h  = anc_h * torch.exp(th)
            
            # clamp => [0..1] (assuming normalized coords)
            pred_cx = pred_cx.clamp(0,1)
            pred_cy = pred_cy.clamp(0,1)
            pred_w  = pred_w.clamp(min=1e-6, max=1.0)
            pred_h  = pred_h.clamp(min=1e-6, max=1.0)

            x1 = pred_cx - 0.5*pred_w
            y1 = pred_cy - 0.5*pred_h
            x2 = pred_cx + 0.5*pred_w
            y2 = pred_cy + 0.5*pred_h
            # shape => [B,nAnchors,4]
            pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            
            # 5) filter + NMS per image
            from torchvision.ops import nms
            all_boxes  = []
            all_labels = []
            all_scores = []
            
            for b_idx in range(B):
                scores_b = cat_prob[b_idx]  # [nAnchors]
                boxes_b  = pred_boxes_xyxy[b_idx]  # [nAnchors,4]
                
                # filter by confidence threshold
                mask = (scores_b > obj_threshold)
                boxes_filt  = boxes_b[mask]
                scores_filt = scores_b[mask]

                # DEBUG: how many boxes survived the threshold for each image
                print(f"[DEBUG inference] b_idx={b_idx}, boxes_filt={len(boxes_filt)}, threshold={obj_threshold}")

                if boxes_filt.numel() == 0:
                    # no detections
                    all_boxes.append(torch.empty((0,4), device=images.device))
                    all_labels.append(torch.empty((0,), dtype=torch.long, device=images.device))
                    all_scores.append(torch.empty((0,), device=images.device))
                    continue
                
                # run NMS
                keep = nms(boxes_filt, scores_filt, nms_threshold)
                boxes_keep  = boxes_filt[keep]
                scores_keep = scores_filt[keep]
                
                # single-class => label=0 => 'cat'
                labels_keep = torch.zeros_like(scores_keep, dtype=torch.long)
                
                all_boxes.append(boxes_keep)
                all_labels.append(labels_keep)
                all_scores.append(scores_keep)
            
            return all_boxes, all_labels, all_scores
