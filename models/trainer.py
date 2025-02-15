# torch
import torch
import random
from torchvision.utils import make_grid

# visualizer
from tqdm import tqdm

# my imports
from utils.plot_utils import scsi_images_bbox_grid

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
        self.stopping_patience = stopping_patience
        
        # train step counter
        self.step_idx = 0

        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        
        # TensorBoard writer
        self.writer = writer
        
        # select image indices for plotting every X epochs
        self.img_idx = random.sample(range(len(self.val_dataloader.dataset)), k=8)

    def train_epoch(self, epoch_idx):
        self.model.train()
        
        # loop over the data
        for images, targets in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch_idx+1}"):
            # organize data
            images = images.to(self.device) # (N, 3, H, W)
            bboxes = targets['boxes'].to(self.device)  # single tensor (N, 1, 4)
            labels = targets['labels'].to(self.device)   # single tensor (N, 1)
            
            # forward pass
            pred_boxes, pred_labels= self.model(images)
            
            # compute loss
            bbox_loss = self.bbox_loss_fn(pred_boxes, bboxes.squeeze(1))
            class_loss = self.class_loss_fn(pred_labels, labels)
            loss = bbox_loss + class_loss
            
            # backward pass + update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # count how many correct predictions were there
            acc = self.compute_predictions(pred_labels)
            batch_acc = (acc == labels).float().sum().item()
            
            # Log losses to TensorBoard
            batch_size = images.shape[0]
            self.writer.add_scalars("Train/Loss", {"BoundingBox": bbox_loss.item() ,"Classification": class_loss.item(), "Total": loss.item()}, self.step_idx)
            self.writer.add_scalar("Train/Acc", batch_acc / batch_size, self.step_idx)
            self.writer.add_scalar("Train/Lr", self.scheduler.get_last_lr()[0], self.step_idx)
            self.step_idx += 1

    def validate_epoch(self, epoch_idx):
        self.model.eval()
        epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc = 0, 0, 0, 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_dataloader,  desc=f"Validation Epoch {epoch_idx+1}"):
                images = images.to(self.device) # (N, 3, H, W)
                bboxes = targets['boxes'].to(self.device)  # single tensor (N, 1, 4)
                labels = targets['labels'].to(self.device)   # single tensor (N, 1)
                
                # predict
                pred_bboxes, pred_labels = self.model(images)
                
                # loss
                bbox_loss = self.bbox_loss_fn(pred_bboxes, bboxes.squeeze(1))
                class_loss = self.class_loss_fn(pred_labels, labels)
                loss = bbox_loss + class_loss
                
                epoch_bbox_loss += bbox_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_total_loss += loss.item()
                
                # compute correct predictions
                predicted = self.compute_predictions(pred_labels)
                epoch_acc += (predicted == labels).float().sum().item()
        
        # average for epoch length
        val_size = len(self.val_dataloader.dataset)
        stats = (epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc)
        epoch_bbox_loss, epoch_class_loss, epoch_total_loss, epoch_acc = (x / val_size for x in stats)
        
        # log
        self.writer.add_scalars("Val/Loss", {"BoundingBox": epoch_bbox_loss ,"Classification": epoch_class_loss, "Total": epoch_total_loss}, self.step_idx)
        self.writer.add_scalar("Val/Acc", epoch_acc, self.step_idx)
        
        return epoch_total_loss

    def train(self, max_epochs, log_images_every = None):
        # early stopping
        best_loss, counter = float('inf'), 0
        
        # start epochs run
        for epoch_idx in range(max_epochs):
            # train
            self.train_epoch(epoch_idx)
            
            # validate
            val_loss = self.validate_epoch(epoch_idx)
            
            # log image results if not None
            if epoch_idx % log_images_every == 0 and log_images_every is not None:
                self.tb_log_images(epoch_idx)
            
            # lr scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # earlys topping mechanism
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

        self.writer.close()
    
    def compute_predictions(self, pred_labels):
        '''counts the number of correct predictions (lables)'''
        return (torch.sigmoid(pred_labels) > 0.5).float() if pred_labels.shape[1] == 1 else pred_labels.argmax(1)
    
    def tb_log_images(self, epoch_idx):
        images_with_boxes = []
        with torch.no_grad():
            # scan the pre-selected random image indices
            for i, idx in enumerate(self.img_idx):
                # get ground truth
                img, target = self.val_dataloader.dataset[idx]  
                img = img.to(self.device).unsqueeze(0)  # add batch dimension
                
                # run model inference
                pred = self.model(img)
                
                # call helper to organize the image
                img_with_boxes = scsi_images_bbox_grid(img, target, pred, self.model.backbone_transforms())
                images_with_boxes.append(img_with_boxes)
                
            # convert to grid row (1 row, N columns)
            img_grid = make_grid(torch.stack(images_with_boxes), nrow=len(images_with_boxes))

            # log to TensorBoard
            self.writer.add_image(f"Img/Val_Results_Epoch_{epoch_idx}", img_grid, epoch_idx)

            # add to tb
            self.writer.add_image(f"Img/Val Results, Epoch = {epoch_idx}", img_grid, epoch_idx)