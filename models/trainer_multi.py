# torch
import torch
import random
from torchvision.utils import make_grid
from torchvision.ops import box_iou, box_convert

# visualizer
from tqdm import tqdm

# my imports
from utils import plot_utils

class TrainerMulti:
    def __init__(self, device, model, optimizer, train_dataloader, val_dataloader, losses, writer, lr_scheduler = None, stopping_patience = None):
        ''' Trainer class.  
            Inputs:
            - device: torch.device
            - model: the model to train
            - optimizer updated with the model parameters
            - dataloaders (Train and validation)
            - loss function: list of two [box_loss, class_loss]
            - the tensorboard writer, initiated outside of the trainer
            - lr_scheduler - optional - lr scheduler wrapping the optimizer.
            - stopping patience - optional - to use early stopping
        '''
        # init
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.epoch = 0
        self.stopping_patience = stopping_patience
        self.num_classes = self.model.num_classes
        
        # train step counter
        self.step_idx = 0

        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        self.obj_loss_fn = losses[2] # for the objectness use the second loss type
        
        # TensorBoard writer
        self.writer = writer
        

    def train_epoch(self, epoch_idx):
        self.model.train()
        
        # loop over the data
        for images, bboxes, labels in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch_idx+1}"):
            # organize data
            images = images.to(self.device) # (B, 3, H, W)
            gt_boxes = [bbox.to(self.device) for bbox in bboxes]
            gt_labels = [label.to(self.device) for label in labels]
            # TODO labels to onehot?
            
            # forward pass
            pred_boxes, pred_labels_logits, pred_obj_logits = self.model(images)
            
            # match boxes to gt
            obj_targets, bbox_targets, cls_targets = self.assign_targets(pred_boxes, gt_boxes, gt_labels, grid_size=7)
            
            # mask according to matches
            obj_mask = (obj_targets.squeeze(-1) == 1)
            
            # Compute bounding box loss only for the selected grid cells
            if obj_mask.sum() > 0: # if there are matches
                # compute bbox loss for the matches
                bbox_loss = self.bbox_loss_fn(pred_boxes[obj_mask], bbox_targets[obj_mask])
                # compute class loss for the matches
                class_loss = self.class_loss_fn(pred_labels_logits[obj_mask], cls_targets[obj_mask])
            else:
                bbox_loss = torch.tensor(0.0, device=self.device)
                class_loss = torch.tensor(0.0, device=self.device)
            
            # compute objectiveness loss
            obj_loss = self.obj_loss_fn(pred_obj_logits.view(-1), obj_targets.view(-1))
            
            # compute loss
            total_loss = bbox_loss + class_loss + obj_loss
            
            # backward pass + update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # compute stats
            batch_acc, batch_iou = self.compute_batch_stats(self.compute_predictions(pred_labels_logits), 
                                                            cls_targets, 
                                                            pred_boxes, 
                                                            bbox_targets, 
                                                            mask=obj_mask)
            
            # Log losses to TensorBoard
            self.writer.add_scalars("Train/Loss", {
                "BoundingBox": bbox_loss.item(),
                "Classification": class_loss.item(),
                "Objectness": obj_loss.item(),
                "Total": total_loss.item()
            }, self.step_idx)
            
            self.writer.add_scalars("Train/Stats", {
                "Acc": batch_acc,
                "IoU": batch_iou
            }, self.step_idx)
            self.writer.add_scalar("Train/Lr", self.scheduler.get_last_lr()[0], self.step_idx)
            self.writer.flush()
            self.step_idx += 1

    def validate_epoch(self, epoch_idx):
        epoch_bbox_loss, epoch_class_loss, epoch_obj_loss, epoch_total_loss = 0, 0, 0, 0
        epoch_acc, epoch_iou = 0, 0        
        
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (images, bboxes, labels) in enumerate(tqdm(self.val_dataloader,  desc=f"Validation Epoch {epoch_idx+1}")):
                images = images.to(self.device) # (B, 3, H, W)
                gt_boxes = [bbox.to(self.device) for bbox in bboxes]
                gt_labels = [label.to(self.device) for label in labels]
                
                # predict using inference!
                boxes, labels, confs = self.model.inference(images)
                
                # TODO - all the computation, really
                
                # compute loss
                # total_loss = bbox_loss + class_loss + obj_loss
                
                # epoch_bbox_loss += bbox_loss.item()
                # epoch_class_loss += class_loss.item()
                # epoch_obj_loss += obj_loss.item()
                # epoch_total_loss += total_loss.item()
                
                # compute stats
                # target_labels_grid = cls_targets.argmax(dim=-1)  # shape: [B,49]
                # batch_acc, batch_iou = self.compute_batch_stats(self.compute_predictions(pred_labels_logits), 
                #                                             target_labels_grid, 
                #                                             pred_boxes, 
                #                                             bbox_targets, 
                #                                             mask=obj_mask)
                
                # epoch_acc += batch_acc
                # epoch_iou += batch_iou
        
        # average for epoch length
        stats = (epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc, epoch_iou)
        epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc, epoch_iou = tuple(x / (batch_idx + 1) for x in stats)
        
        # log
        self.writer.add_scalars("Val/Loss", {"BoundingBox": epoch_bbox_loss ,"Classification": epoch_class_loss, "Total": epoch_total_loss}, self.step_idx)
        self.writer.add_scalars("Val/stats", {"Acc": epoch_acc, "IoU": epoch_iou}, self.step_idx)
        
        return epoch_total_loss

    def train(self, max_epochs, log_images_every = None, img_plot_qty = 12, mask_single_object = False):
        self.mask_single_object = mask_single_object
        
        # early stopping
        best_loss, counter = float('inf'), 0
        
        #  image logging params
        if log_images_every is not None:
            # select image indices for plotting every X epochs
            self.img_idx = random.sample(range(len(self.val_dataloader.dataset)), k=img_plot_qty)
        
        # start epochs run
        for epoch in range(self.epoch, self.epoch + max_epochs):
            # train
            self.epoch = epoch
            self.train_epoch(self.epoch)
            
            # validate
            val_loss = self.validate_epoch(self.epoch)
            
            # log image results if not None
            if self.epoch % log_images_every == 0 and log_images_every is not None:
                self.tb_log_voc_images(self.epoch)
            
            # lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # early stopping mechanism
            if self.stopping_patience is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    # check to see if it's time for early stopping
                    if counter >= self.stopping_patience:
                        print("Early stopping triggered. Training terminated.")
                        break
        
        # log last image grid and return
        return self.tb_log_voc_images(self.epoch)
    
    
    def assign_targets(self, pred_boxes, gt_boxes, gt_labels, grid_size = 7):
        '''Matches bboxes with the closes predicted bbox.
        Args:
            pred_boxes: [B,49,4]
            gt_boxes: [B,N,4]
            gt_labels: [B,N]
            grid_size: the feature map output shape
        Returns:
            obj_targets:[B,49,1] - the objectness ground truth
            bbox_targets:[B,49,4] - gt target for each pred bbox
            cls_targets: [B,49,1] - gt label for each pred label, in one-hot encoding
            '''
            
        B = pred_boxes.size(0)
        
        obj_targets = torch.zeros(B, grid_size ** 2, 1, device = self.device)
        bbox_targets = torch.zeros_like(pred_boxes)
        cls_targets = torch.zeros(B, grid_size ** 2, 1, device = self.device)
        
        # loop over images
        for b in range(B):
            # get the number of gt objects in the bbox
            N = gt_boxes[b].size(0)
            
            # scan each object
            for i in range(N):
                # compute the bbox center assuming the box is xywh
                center_x = gt_boxes[b][i][0] + gt_boxes[b][i][2] / 2.0
                center_y = gt_boxes[b][i][1] + gt_boxes[b][i][3] / 2.0
                
                # find the closest grid point
                col = int(center_x * grid_size)
                row = int(center_y * grid_size)
                
                # Make sure indices are within bounds.
                col = min(col, grid_size - 1)
                row = min(row, grid_size - 1)
                grid_idx = row * grid_size + col
                
                # Set the objectness target to 1 for the chosen grid cell.
                obj_targets[b, grid_idx, 0] = 1.0
                
                # Set the bbox target for this grid cell to be the ground truth bbox.
                bbox_targets[b, grid_idx] = gt_boxes[b][i]
                
                # handle labels
                # one_hot = torch.zeros(self.num_classes, device=self.device)
                # mark the one hot vector according to the label
                # one_hot[int(gt_labels[b][i])] = 1.0
                # cls_targets[b, grid_idx] = one_hot
                cls_targets[b,grid_idx] = gt_labels[b][i]
                
        return obj_targets, bbox_targets, cls_targets
    
    def compute_predictions(self, logits_labels):
        if logits_labels.shape[-1] == 1:
            return (torch.sigmoid(logits_labels) > 0.5).float()
        else:
            return logits_labels.argmax(dim=-1)

    
    def compute_batch_stats(self, pred_labels, target_labels, pred_boxes, target_boxes, mask = None):
        '''computes accuracy and IoU for a batch.'''
        
        # if we compute only for the mask
        if mask is not None:
            pred_labels = pred_labels[mask]
            target_labels = target_labels[mask]
            pred_boxes = pred_boxes[mask]
            target_boxes = target_boxes[mask]
        
        # compute accuracy
        predicted_correct = (pred_labels == target_labels).float().sum().item()
        total = target_labels.numel() if target_labels.numel() > 0 else 1
        accuracy = predicted_correct / total

        # compute IoU
        if pred_boxes.numel() > 0 and target_boxes.numel() > 0:
            iou_matrix = box_iou(
                box_convert(pred_boxes.view(-1, 4), 'xywh', 'xyxy'),
                box_convert(target_boxes.view(-1, 4), 'xywh', 'xyxy')
            )
            batch_iou = iou_matrix.diag()  # one-to-one correspondence
            mean_iou = batch_iou.mean().item() if batch_iou.numel() > 0 else 0.0
        else:
            mean_iou = 0.0
        
        return accuracy, mean_iou
    
    def tb_log_voc_images(self, epoch_idx):
        images_with_boxes = []
        with torch.no_grad():
            # scan the pre-selected random image indices
            for idx in self.img_idx:
                # get ground truth
                img, bboxes, labels  = self.val_dataloader.dataset[idx]  
                img = img.to(self.device).unsqueeze(0)  # add batch dimension
                bboxes = bboxes.to(self.device)
                labels = labels.to(self.device)
                
                # run model inference and convert to probabilities
                pred_bboxes, pred_labels, _ = self.model.inference(img, obj_threshold=0.5, nms_threshold=0.5)
                
                # un-normalize image
                mean, std = self.model.backbone_transforms().mean, self.model.backbone_transforms().std
                img = plot_utils.unnormalize(img, mean, std)
                
                # plot bbox of target and prediction with labels
                img_with_boxes = plot_utils.voc_img_bbox_plot(img.squeeze(0), bboxes, labels, pred_bboxes.squeeze(0), pred_labels.squeeze(0))
                images_with_boxes.append(img_with_boxes)
            
            # convert to grid row (1 row, N columns)
            img_grid = make_grid(torch.stack(images_with_boxes), nrow=6)
            
            # log to TensorBoard
            self.writer.add_image(f"Img/Val_Results_Epoch_{epoch_idx}", img_grid, epoch_idx)
            
            return img_grid