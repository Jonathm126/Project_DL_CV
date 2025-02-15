# torch
import torch
from torch.utils.tensorboard import SummaryWriter

# visualizer
from tqdm import tqdm


class Trainer:
    def __init__(self, device, model, optimizer, train_dataloader, val_dataloader, losses, lr_scheduler = None, stopping_patience = None):
        ''' Trainer class.  
            Inputs:
            - device: torch.device
            - optimizer updated with the model parameters
            - dataloaders 
            - loss function: list of two [box_loss, class_loss]
            - lr_scheduler - optional - lr scheduler wrapping the optimizer.
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
        self.writer = SummaryWriter()
        # add the model graph to the writer
        sim_image = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32).to(device)
        self.writer.add_graph(model, sim_image)

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
            
            # average stats for batch length
            batch_size = images.shape[0]
            stats = (bbox_loss.item(), class_loss.item(), loss.item(), batch_acc)
            batch_bbox_loss, batch_class_loss, batch_loss, batch_acc = (x / batch_size for x in stats)
            
            # Log losses to TensorBoard
            self.step_idx += 1
            self.writer.add_scalars("Train/Loss", {"BoundingBox": batch_bbox_loss ,"Classification": batch_class_loss, "Total": batch_loss}, self.step_idx)
            self.writer.add_scalar("Train/Acc", batch_acc, self.step_idx)
            self.writer.add_scalar("Train/Lr", self.scheduler.get_last_lr()[0], self.step_idx)

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

    def train(self, max_epochs):
        # start epochs run
        for epoch_idx in range(max_epochs):
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