import torch
import random
import math
from torchvision.utils import make_grid
from torchvision.ops import box_iou
from tqdm import tqdm

from utils import plot_utils

class TrainerMulti:
    def __init__(
        self, 
        device, 
        model, 
        optimizer, 
        train_dataloader, 
        val_dataloader, 
        losses, 
        writer, 
        lr_scheduler=None, 
        stopping_patience=None
    ):
        """
        - device: Torch device
        - model: MoMiDetectionModel
        - train_dataloader, val_dataloader
        - losses: [bbox_loss_fn, class_loss_fn, (optional) obj_loss_fn]
        - writer: TensorBoard writer
        - lr_scheduler: optional
        - stopping_patience: optional early stopping
        """
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.epoch = 0
        self.stopping_patience = stopping_patience
        self.num_classes = self.model.num_classes  # e.g. 1 => cat only
        self.step_idx = 0

        # We assume: losses = [bbox_loss_fn, class_loss_fn, obj_loss_fn(optional)]
        self.bbox_loss_fn = losses[0]
        self.class_loss_fn = losses[1]
        if len(losses) > 2:
            self.obj_loss_fn = losses[2]
        else:
            self.obj_loss_fn = None
        
        self.writer = writer
    
    def train(self, max_epochs, log_images_every=None, img_plot_qty=12):
        best_loss = float('inf')
        counter = 0
        
        if log_images_every is not None:
            self.img_idx = random.sample(range(len(self.val_dataloader.dataset)), k=img_plot_qty)
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            
            if log_images_every is not None and (epoch % log_images_every == 0):
                self.tb_log_voc_images(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # simple early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if self.stopping_patience is not None and counter >= self.stopping_patience:
                    print("Early stopping triggered.")
                    break
        
        final_img_grid = self.tb_log_voc_images(self.epoch)
        return final_img_grid

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_steps = 0
        running_loss = 0.0
        
        for batch_idx, (images, bboxes, labels) in enumerate(
            tqdm(self.train_dataloader, desc=f"Train Epoch {epoch_idx+1}")
        ):
            images = images.to(self.device)
            gt_boxes_list = [b.to(self.device) for b in bboxes]
            gt_labels_list = [l.to(self.device) for l in labels]
            
            # forward
            pred_offsets, pred_cls_logits = self.model(images)
            # shape => [B, 49*k, 4], [B, 49*k, 2]
            
            # DEBUG: print max cat prob before we do any anchor assignment
            with torch.no_grad():
                # shape => [B,49*k,2]
                probs = torch.softmax(pred_cls_logits, dim=-1)
                cat_prob = probs[...,1]  # cat channel
                cat_prob_max = cat_prob.max().item()
                print(f"[DEBUG train_epoch] Batch={batch_idx}, cat_prob.max()={cat_prob_max:.4f}")

            obj_targets, offsets_targets, class_targets = self.assign_targets(
                pred_offsets, gt_boxes_list, gt_labels_list
            )
            # obj_targets => [B,49*k], 1 => object anchor, 0 => background
            # offsets_targets => [B,49*k,4]
            # class_targets => [B,49*k], label=1 => cat, 0 => BG
            
            B, NA, _ = pred_offsets.shape
            pred_offsets_flat = pred_offsets.view(B * NA, 4)
            pred_cls_logits_flat = pred_cls_logits.view(B * NA, 2)  # expecting 2 => (bg, cat)
            
            obj_targets_flat = obj_targets.view(-1)
            offsets_targets_flat = offsets_targets.view(B * NA, 4)
            class_targets_flat = class_targets.view(-1)
            
            # bounding box loss => only for anchors with obj=1
            obj_mask = (obj_targets_flat > 0.5)
            
            # DEBUG: check how many positives in this batch
            num_pos = obj_mask.sum().item()
            print(f"[DEBUG train_epoch] Batch={batch_idx}, positives={num_pos}")

            if obj_mask.sum() > 0:
                bbox_loss = self.bbox_loss_fn(
                    pred_offsets_flat[obj_mask], offsets_targets_flat[obj_mask]
                )
            else:
                bbox_loss = torch.tensor(0.0, device=self.device)
            
            # classification => cross entropy (cat vs bg)
            class_loss = self.class_loss_fn(pred_cls_logits_flat, class_targets_flat.long())
            
            total_loss = bbox_loss + class_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            total_steps += 1
            self.writer.add_scalar("Train/Total_Loss", total_loss.item(), self.step_idx)
            self.step_idx += 1
        
        epoch_loss = running_loss / total_steps
        print(f"Epoch {epoch_idx+1} => training loss= {epoch_loss:.4f}")

    def validate_epoch(self, epoch_idx):
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for images, bboxes, labels in self.val_dataloader:
                images = images.to(self.device)
                total_steps += 1
        
        val_loss = total_loss / max(1, total_steps)
        print(f"Epoch {epoch_idx+1} => validation loss= {val_loss:.4f}")
        return val_loss
    
    def assign_targets(self, pred_offsets, gt_boxes_list, gt_labels_list, iou_thresh=0.3):
        B, NA, _ = pred_offsets.shape
        device = pred_offsets.device
        
        base_anchors = []
        grid_size = self.model.feature_map_size
        for row in range(grid_size):
            for col in range(grid_size):
                cx = (col+0.5)/grid_size
                cy = (row+0.5)/grid_size
                for (scale, ar) in self.model.anchors:
                    w = scale * math.sqrt(ar)
                    h = scale / math.sqrt(ar)
                    base_anchors.append([cx, cy, w, h])
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        
        acx = base_anchors[:,0]
        acy = base_anchors[:,1]
        aw  = base_anchors[:,2]
        ah  = base_anchors[:,3]
        ax1 = acx - 0.5*aw
        ay1 = acy - 0.5*ah
        ax2 = acx + 0.5*aw
        ay2 = acy + 0.5*ah
        anchors_xyxy = torch.stack([ax1, ay1, ax2, ay2], dim=-1)
        
        obj_targets = torch.zeros(B, NA, device=device)
        offsets_targets = torch.zeros(B, NA, 4, device=device)
        class_targets = torch.zeros(B, NA, device=device)  # 0 => BG, 1 => cat
        
        from torchvision.ops import box_iou
        for b_idx in range(B):
            gt_boxes  = gt_boxes_list[b_idx]
            gt_labels = gt_labels_list[b_idx]
            if gt_boxes.numel() == 0:
                continue  # no GT => skip
            
            # convert GT to xyxy
            gx1 = gt_boxes[:, 0]
            gy1 = gt_boxes[:, 1]
            gx2 = gx1 + gt_boxes[:, 2]
            gy2 = gy1 + gt_boxes[:, 3]
            gt_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=-1)
            
            ious = box_iou(anchors_xyxy, gt_xyxy)  # [NA, N]
            iou_vals, anchor_idx = ious.max(dim=0)  # best anchor per GT
            
            for i, iouVal in enumerate(iou_vals):
                if iouVal > iou_thresh:
                    bestA = anchor_idx[i].item()
                    obj_targets[b_idx, bestA] = 1
                    class_targets[b_idx, bestA] = 1  # cat
                    # offsets
                    anc_cx, anc_cy, anc_w, anc_h = base_anchors[bestA]
                    gt_x, gt_y, gt_w, gt_h = gt_boxes[i]
                    gt_cx = gt_x + 0.5*gt_w
                    gt_cy = gt_y + 0.5*gt_h
                    tx = (gt_cx - anc_cx)/anc_w
                    ty = (gt_cy - anc_cy)/anc_h
                    tw = torch.log(gt_w/anc_w + 1e-8)
                    th = torch.log(gt_h/anc_h + 1e-8)
                    offsets_targets[b_idx, bestA] = torch.stack([tx,ty,tw,th], dim=0)
        
        return obj_targets, offsets_targets, class_targets.unsqueeze(-1)

    def tb_log_voc_images(self, epoch_idx):
        """
        Log some validation images + predicted boxes to TensorBoard.
        We lower the obj_threshold to 0.2 so we can see more boxes,
        and print how many boxes we found per image.
        """
        images_with_boxes = []
        with torch.no_grad():
            self.model.eval()
            if not hasattr(self, 'img_idx'):
                return None
            
            for idx in self.img_idx:
                img, bboxes, labels = self.val_dataloader.dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                bboxes = bboxes.to(self.device)
                labels = labels.to(self.device)
                
                # lower threshold to 0.2
                all_boxes, all_lbls, all_scores = self.model.inference(img, obj_threshold=0.2, nms_threshold=0.4)
                
                # each is list of length B=1 => all_boxes[0], ...
                pred_bboxes = all_boxes[0]
                pred_labels = all_lbls[0]
                pred_scores = all_scores[0]
                
                print(f"[tb_log_voc_images] idx={idx} => Found {len(pred_bboxes)} boxes, scores={pred_scores.cpu().numpy()}")
                
                mean, std = self.model.backbone_transforms().mean, self.model.backbone_transforms().std
                unnorm_img = plot_utils.unnormalize(img, mean, std)
                
                img_with_boxes = plot_utils.voc_img_bbox_plot(
                    unnorm_img.squeeze(0), 
                    bboxes, labels, 
                    pred_bboxes, pred_labels
                )
                images_with_boxes.append(img_with_boxes)
            
            if len(images_with_boxes) > 0:
                from torchvision.utils import make_grid
                grid = make_grid(torch.stack(images_with_boxes), nrow=4)
                self.writer.add_image(f"ValResults/Epoch_{epoch_idx}", grid, epoch_idx)
                return grid
        return None
