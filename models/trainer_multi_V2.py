import torch
import random
import math
from torchvision.utils import make_grid
from torchvision.ops import box_iou
from tqdm import tqdm

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
        Args:
            device: torch device (cuda, mps, or cpu)
            model: your MoMiDetectionModel that returns (bbox_out, cls_out)
            optimizer: e.g. Adam or SGD
            train_dataloader, val_dataloader: DataLoader for train/val
            losses: [bbox_loss_fn, class_loss_fn, (optional) obj_loss_fn]
            writer: TensorBoard SummaryWriter
            lr_scheduler: optional learning-rate scheduler
            stopping_patience: optional early-stopping patience
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

        # losses = [bbox_loss_fn, class_loss_fn, (optional) obj_loss_fn]
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
        
        # For TensorBoard image-logging
        if log_images_every is not None:
            # pick random val-dataset indices to visualize
            self.img_idx = random.sample(
                range(len(self.val_dataloader.dataset)), 
                k=min(img_plot_qty, len(self.val_dataloader.dataset))
            )
        
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
            
            # Forward pass => 2 outputs
            pred_offsets, pred_cls_logits = self.model(images)

            with torch.no_grad():
                # cat_prob => softmax along last dim => index=1 is 'cat'
                cat_prob = torch.softmax(pred_cls_logits, dim=-1)[:,:,1]  
                max_cat_prob = cat_prob.max().item()
            
            # Match anchor targets
            obj_targets, offsets_targets, class_targets = self.assign_targets(
                pred_offsets, gt_boxes_list, gt_labels_list
            )
            positives_count = (obj_targets > 0.5).sum().item()

            # Some debugging
            print(f"[DEBUG train_epoch] Batch={batch_idx}, cat_prob.max()={max_cat_prob:.4f}")
            print(f"[DEBUG train_epoch] Batch={batch_idx}, positives={positives_count}")

            # Flatten predictions & targets
            B, NA, _ = pred_offsets.shape
            pred_offsets_flat = pred_offsets.view(B * NA, 4)
            pred_cls_logits_flat = pred_cls_logits.view(B * NA, 2)

            obj_targets_flat = obj_targets.view(-1)
            offsets_targets_flat = offsets_targets.view(B * NA, 4)
            class_targets_flat = class_targets.view(-1)

            # bounding box loss => only for anchors with obj=1
            obj_mask = (obj_targets_flat > 0.5)
            if obj_mask.sum() > 0:
                bbox_loss = self.bbox_loss_fn(
                    pred_offsets_flat[obj_mask], 
                    offsets_targets_flat[obj_mask]
                )
            else:
                bbox_loss = torch.tensor(0.0, device=self.device)
            
            # classification => cat vs bg => cross entropy
            class_loss = self.class_loss_fn(
                pred_cls_logits_flat, 
                class_targets_flat.long()
            )

            # If you wanted a separate obj_loss, you could add it here
            # but typically "obj-ness" is the cat vs. bg classification
            if self.obj_loss_fn is not None:
                # For example, you could treat obj_targets as 1/0
                # But that would be double-counting the classification
                obj_loss = self.obj_loss_fn(
                    pred_cls_logits_flat[:,1],  # just the 'cat' logit
                    obj_targets_flat
                )
                total_loss = bbox_loss + class_loss + obj_loss
            else:
                total_loss = bbox_loss + class_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            running_loss += total_loss.item()
            total_steps += 1
            self.writer.add_scalar("Train/Total_Loss", total_loss.item(), self.step_idx)
            self.step_idx += 1
        
        epoch_loss = running_loss / max(1, total_steps)
        print(f"Epoch {epoch_idx+1} => training loss= {epoch_loss:.4f}")

    def validate_epoch(self, epoch_idx):
        """
        Minimal validation loop.
        You can expand this to compute actual validation losses if needed.
        Right now, it just prints a dummy 0.0 for 'val_loss'.
        """
        self.model.eval()
        total_steps = 0
        
        # If you want a real validation loss, do a forward pass, compute losses, etc.
        # For now, we'll just count how many val batches there are.
        with torch.no_grad():
            for _batch_idx, (images, bboxes, labels) in enumerate(self.val_dataloader):
                total_steps += 1
        
        # Just for demonstration
        val_loss = 0.0
        print(f"Epoch {epoch_idx+1} => validation loss= {val_loss:.4f}")
        return val_loss
    
    def assign_targets(self, pred_offsets, gt_boxes_list, gt_labels_list, iou_thresh=0.5):
        """
        Two-pass anchor matching:
        1) For each anchor, find its best GT. If IoU > iou_thresh => assign positive.
        2) For each GT, find its best anchor => also assign positive
        """
        B, NA, _ = pred_offsets.shape
        device = pred_offsets.device
        
        # 1) build base anchors from your model
        grid_size = self.model.feature_map_size
        base_anchors = []
        for row in range(grid_size):
            for col in range(grid_size):
                cx = (col+0.5)/grid_size
                cy = (row+0.5)/grid_size
                for (scale, ar) in self.model.anchors:
                    w = scale * math.sqrt(ar)
                    h = scale / math.sqrt(ar)
                    base_anchors.append([cx, cy, w, h])
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
        
        # convert anchors to xyxy
        acx = base_anchors[:,0]
        acy = base_anchors[:,1]
        aw  = base_anchors[:,2]
        ah  = base_anchors[:,3]
        ax1 = acx - 0.5*aw
        ay1 = acy - 0.5*ah
        ax2 = acx + 0.5*aw
        ay2 = acy + 0.5*ah
        anchors_xyxy = torch.stack([ax1, ay1, ax2, ay2], dim=-1)  # [NA,4]
        
        obj_targets     = torch.zeros(B, NA, device=device)
        offsets_targets = torch.zeros(B, NA, 4, device=device)
        class_targets   = torch.zeros(B, NA, device=device)  # 0 => BG, 1 => cat
        
        for b_idx in range(B):
            gt_boxes  = gt_boxes_list[b_idx]
            gt_labels = gt_labels_list[b_idx]
            if gt_boxes.numel() == 0:
                # no GT => all background
                continue
            
            # convert GT from (x,y,w,h) to xyxy
            gx1 = gt_boxes[:, 0]
            gy1 = gt_boxes[:, 1]
            gx2 = gx1 + gt_boxes[:, 2]
            gy2 = gy1 + gt_boxes[:, 3]
            gt_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=-1)  # [num_gt,4]

            # compute IoU => shape [NA, num_gt]
            ious = box_iou(anchors_xyxy, gt_xyxy)
            num_gt = gt_xyxy.size(0)
            
            # PASS A: For each anchor => pick best GT
            best_iou_for_anchor, best_gt_idx_for_anchor = ious.max(dim=1)
            anchor_mask = (best_iou_for_anchor > iou_thresh)
            assigned_idxs = anchor_mask.nonzero(as_tuple=True)[0]
            for a_idx in assigned_idxs:
                gt_idx = best_gt_idx_for_anchor[a_idx]
                obj_targets[b_idx, a_idx] = 1
                class_targets[b_idx, a_idx] = 1  # single-class => cat
                # compute offsets
                anc_cx, anc_cy, anc_w, anc_h = base_anchors[a_idx]
                gt_x, gt_y, gt_w, gt_h = gt_boxes[gt_idx]
                
                gt_cx = gt_x + 0.5 * gt_w
                gt_cy = gt_y + 0.5 * gt_h
                tx = (gt_cx - anc_cx) / anc_w
                ty = (gt_cy - anc_cy) / anc_h
                tw = torch.log(gt_w/anc_w + 1e-8)
                th = torch.log(gt_h/anc_h + 1e-8)
                offsets_targets[b_idx, a_idx] = torch.tensor([tx, ty, tw, th], device=device)
            
            # PASS B: For each GT => pick best anchor
            best_iou_for_gt, anchor_idx_for_gt = ious.max(dim=0)  # shape [num_gt]
            for gt_i in range(num_gt):
                a_idx   = anchor_idx_for_gt[gt_i]
                obj_targets[b_idx, a_idx] = 1
                class_targets[b_idx, a_idx] = 1
                anc_cx, anc_cy, anc_w, anc_h = base_anchors[a_idx]
                gt_x, gt_y, gt_w, gt_h = gt_boxes[gt_i]
                
                gt_cx = gt_x + 0.5 * gt_w
                gt_cy = gt_y + 0.5 * gt_h
                tx = (gt_cx - anc_cx) / anc_w
                ty = (gt_cy - anc_cy) / anc_h
                tw = torch.log(gt_w/anc_w + 1e-8)
                th = torch.log(gt_h/anc_h + 1e-8)
                offsets_targets[b_idx, a_idx] = torch.tensor([tx, ty, tw, th], device=device)
        
        # return shapes: [B,NA], [B,NA,4], [B,NA,1]
        return obj_targets, offsets_targets, class_targets.unsqueeze(-1)

    def tb_log_voc_images(self, epoch_idx):
        """
        Log some validation images + predicted boxes to TensorBoard.
        We lower the obj_threshold so we can see more boxes,
        and print how many boxes we found per image.
        """
        images_with_boxes = []
        with torch.no_grad():
            self.model.eval()
            if not hasattr(self, 'img_idx'):
                return None
            
            for idx in self.img_idx:
                # get the sample from the dataset
                img, bboxes, labels = self.val_dataloader.dataset[idx]
                img = img.unsqueeze(0).to(self.device)
                bboxes = bboxes.to(self.device)
                labels = labels.to(self.device)
                
                # run inference
                all_boxes, all_lbls, all_scores = self.model.inference(
                    img, obj_threshold=0.2, nms_threshold=0.4
                )
                pred_bboxes = all_boxes[0]
                pred_labels = all_lbls[0]
                pred_scores = all_scores[0]
                
                print(f"[tb_log_voc_images] idx={idx} => Found {len(pred_bboxes)} boxes, scores={pred_scores.cpu().numpy()}")
                
                # Unnormalize the image
                mean, std = self.model.backbone_transforms().mean, self.model.backbone_transforms().std
                unnorm_img = self._unnormalize(img, mean, std)
                
                # Draw them => red=GT, blue=pred, etc. (stub)
                img_with_boxes = self._plot(
                    unnorm_img.squeeze(0), bboxes, labels, 
                    pred_bboxes, pred_labels
                )
                images_with_boxes.append(img_with_boxes)
            
            if len(images_with_boxes) > 0:
                grid = make_grid(torch.stack(images_with_boxes), nrow=4)
                self.writer.add_image(f"ValResults/Epoch_{epoch_idx}", grid, epoch_idx)
                return grid
        return None

    def _unnormalize(self, x, mean, std):
        for c in range(x.shape[1]):
            x[:, c, :, :] = x[:, c, :, :] * std[c] + mean[c]
        return x

    def _plot(self, image_tensor, bboxes, labels, pred_bboxes, pred_labels):
        """
        Placeholder for bounding-box plotting.
        Return a (C,H,W) tensor image with boxes drawn. 
        """
        # If you have your own utils, call them; otherwise do minimal logic
        return image_tensor
