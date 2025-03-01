# torch
import torch
import random
from torchvision.utils import make_grid
from torchvision.ops import box_iou, box_convert

# visualizer
from tqdm import tqdm

# my imports
from utils import plot_utils

class Trainer:
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
        
        # train step counter
        self.step_idx = 0

        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        
        # TensorBoard writer
        self.writer = writer
        

    def train_epoch(self, epoch_idx):
        self.model.train()
        
        # loop over the data
        for images, bboxes, labels  in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch_idx+1}"):
            # organize data
            images = images.to(self.device) # (N, 3, H, W)
            bboxes = bboxes.to(self.device)  # single tensor (N, 1, 4)
            labels = labels.to(self.device)   # single tensor (N, 1)
            
            # forward pass
            pred_boxes, pred_labels_logits = self.model(images)
            
            # compute loss
            bbox_loss = self.bbox_loss_fn(pred_boxes, bboxes)#, reduction = 'mean'
            class_loss = self.class_loss_fn(pred_labels_logits, labels)
            loss =  bbox_loss + class_loss
            
            # backward pass + update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # compute stats
            pred_labels = self.compute_predictions(pred_labels_logits)
            batch_acc, batch_iou = self.compute_batch_stats(pred_labels, labels, pred_boxes, bboxes)
            
            # Log losses to TensorBoard
            self.writer.add_scalars("Train/Loss", {"BoundingBox": bbox_loss.item() ,"Classification": class_loss.item(), "Total": loss.item()}, self.step_idx)
            self.writer.add_scalars("Train/Stats", {"Acc" : batch_acc, "IoU" : batch_iou}, self.step_idx)
            if self.scheduler is not None:
                self.writer.add_scalar("Train/Lr", self.scheduler.get_last_lr()[0], self.step_idx)
            self.step_idx += 1

    def validate_epoch(self, epoch_idx):
        epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc, epoch_iou = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (images, bboxes, labels) in enumerate(tqdm(self.val_dataloader,  desc=f"Validation Epoch {epoch_idx+1}")):
                images = images.to(self.device) # (B, 3, H, W)
                bboxes = bboxes.to(self.device)  # single tensor (B, N, 4)
                labels = labels.to(self.device)   # single tensor (B, N)
                
                # predict
                pred_bboxes, pred_labels_logits = self.model(images)
                
                # loss
                bbox_loss = self.bbox_loss_fn(pred_bboxes, bboxes) #, reduction = 'mean'
                class_loss = self.class_loss_fn(pred_labels_logits, labels)
                loss = bbox_loss + class_loss
                
                epoch_bbox_loss += bbox_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_total_loss += loss.item()
                
                # compute stats
                pred_labels = self.compute_predictions(pred_labels_logits)
                batch_acc, batch_iou = self.compute_batch_stats(pred_labels, labels, pred_bboxes, bboxes)
                
                epoch_acc += batch_acc
                epoch_iou += batch_iou
        
        # average for epoch length
        stats = (epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc, epoch_iou)
        epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc, epoch_iou = tuple(x / (batch_idx + 1) for x in stats)
        
        # log
        self.writer.add_scalars("Val/Loss", {"BoundingBox": epoch_bbox_loss ,"Classification": epoch_class_loss, "Total": epoch_total_loss}, self.step_idx)
        self.writer.add_scalars("Val/stats", {"Acc": epoch_acc, "IoU": epoch_iou}, self.step_idx)
        
        return epoch_total_loss

    def train(self, max_epochs, log_images_every = None, img_plot_qty = 12):
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
    
    def compute_predictions(self, logits_labels):
        '''converts the logits to label predictions'''
        return (torch.sigmoid(logits_labels) > 0.5).float() if logits_labels.shape[1] == 1 else logits_labels.argmax(1)
    
    def compute_batch_stats(self, pred_labels, labels, pred_boxes, boxes):
        '''computes accuracy and IoU for a batch.'''
        # compute accuracy
        predicted = (pred_labels == labels).float().sum().item()
        accuracy = predicted / len(labels)

        # compute IoU
        # cast to xyxy
        iou_matrix = box_iou(box_convert(pred_boxes.view(-1, 4), 'xywh', 'xyxy'),
                            box_convert(boxes.view(-1, 4), 'xywh', 'xyxy'))
        batch_iou = iou_matrix.diag()  # Get the IoU of each bbox with itself
        mean_iou = batch_iou.mean().item()  # Compute batch mean IoU
        
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
                pred_bboxes, pred_labels = self.model(img)
                pred_labels = self.compute_predictions(pred_labels)
                
                # un-normalize image
                mean, std = self.model.backbone_transforms().mean, self.model.backbone_transforms().std
                img = plot_utils.unnormalize(img, mean, std)
                
                # plot bbox of target and prediction with labels
                img_with_boxes = plot_utils.voc_img_bbox_plot(img.squeeze(0), bboxes, labels, pred_bboxes.squeeze(0), pred_labels.squeeze(1))
                images_with_boxes.append(img_with_boxes)
            
            # convert to grid row (1 row, N columns)
            img_grid = make_grid(torch.stack(images_with_boxes), nrow=6)
            
            # log to TensorBoard
            self.writer.add_image(f"Img/Val_Results_Epoch_{epoch_idx}", img_grid, epoch_idx)
            
            return img_grid