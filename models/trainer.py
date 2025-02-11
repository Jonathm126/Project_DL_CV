# torch
import torch
from torch.utils.tensorboard import SummaryWriter

# visualizer
from tqdm import tqdm


class Trainer:
    def __init__(self, device, model, optimizer, train_dataloader, val_dataloader, losses, max_epochs, lr_scheduler = None, stopping_patience = None):
        ''' Trainer class.  
            Inputs:
            - device: torch.device
            - optimizer updated with the model parameters
            - dataloaders 
            - loss function: list of two [box_loss, class_loss]
            - max epochs (int)
            - lr_scheduler - optional - lr scheduler wrapping the optimizer.
        '''
        # init
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.stopping_patience = stopping_patience

        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        
        # TensorBoard writer
        self.writer = SummaryWriter()

    def train_epoch(self, epoch_idx):
        self.model.train()
        epoch_bbox_loss, epoch_class_loss, train_correct = 0, 0, 0,
        
        # loop over the data
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}/{self.max_epochs}")):
            # organize data
            images = images.to(self.device) # (N, 3, H, W)
            bboxes = targets['boxes'].to(self.device)  # single tensor (N, 1, 4)
            labels = targets['labels'].to(self.device)   # single tensor (N, 1)
            
            # forward pass
            pred_boxes, pred_labels= self.model(images)
            
            # log stats
            bbox_loss = self.bbox_loss_fn(pred_boxes, bboxes.squeeze(1))
            class_loss = self.class_loss_fn(pred_labels, labels)
            
            # the total loss
            loss = bbox_loss + class_loss
            
            # add to epoch loss
            epoch_bbox_loss += bbox_loss.item()
            epoch_class_loss += class_loss.item()
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # count how many correct predictions were there
            predicted = self.compute_predictions(pred_labels)
            train_correct += (predicted == labels).float().sum().item()
            
            # Log losses to TensorBoard
            step = epoch_idx * len(self.val_dataloader) + batch_idx
            self.writer.add_scalars("Loss/Train", {"BoundingBox": bbox_loss.item() ,"Classification": class_loss.item(), "Total": loss.item()}, step)
        
        return epoch_bbox_loss, epoch_class_loss, train_correct

    def validate_epoch(self, epoch_idx):
        self.model.eval()
        epoch_bbox_loss, epoch_class_loss, total_loss, val_correct = 0, 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_dataloader)):
                images = images.to(self.device) # (N, 3, H, W)
                bboxes = targets['boxes'].to(self.device)  # single tensor (N, 1, 4)
                labels = targets['labels'].to(self.device)   # single tensor (N, 1)
                
                # predict
                pred_bboxes, pred_labels = self.model(images)
                
                # loss
                bbox_loss = self.bbox_loss_fn(pred_bboxes, bboxes)
                class_loss = self.class_loss_fn(pred_labels, labels)
                loss = bbox_loss + class_loss
                
                epoch_bbox_loss += bbox_loss.item()
                epoch_class_loss += class_loss.item()
                total_loss += loss.item()
                
                # compute correct predictions
                predicted = self.compute_predictions(pred_labels)
                val_correct += (predicted == labels).float().sum().item()
                
                # log to tensorboard
                step = epoch_idx * len(self.val_dataloader) + batch_idx
                self.writer.add_scalars("Loss/Val", {"BoundingBox": bbox_loss ,"Classification": class_loss, "Total": total_loss}, step)
        
        return epoch_bbox_loss, epoch_class_loss, val_correct

    def train(self):
        # start epochs run
        for epoch_idx in range(self.max_epochs):
            # train
            self.train_epoch(epoch_idx)
            # validate
            val_loss = self.validate_epoch(epoch_idx)
            # lr schedule
            if self.scheduler:
                self.scheduler.step()
            # early stopping
            best_loss, counter = float('inf'), 0
            # check to see if we improved
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