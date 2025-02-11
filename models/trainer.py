# torch
import torch
import torch.nn as nn
import torch.optim as optim

# visualizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, device, optimizer, train_dataloader, val_dataloader, losses, max_epochs = 10, lr_scheduler = None):
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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        
        # scheduler
        if lr_scheduler is not None:
            self.scheudler = lr_scheduler
    
        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        
        # TensorBoard writer
        self.writer = SummaryWriter()
        
    def single_epoch(self, epoch_idx):
        self.model.train()
        epoch_box_loss, epoch_class_loss, trainCorrect = 0, 0, 0 
        
        # loop over the data
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}/{self.max_epochs}")):
            # organize data
            images = images.to(self.device) # (N, 3, H, W)
            boxes = targets['boxes'].to(self.device)  # single tensor (N, 1, 4)
            labels = targets['labels'].to(self.device)   # single tensor (N, 1)
            
            # forward pass
            pred_boxes, pred_labels= self.model(images)
            boxes_loss = self.bbox_loss_fn(pred_boxes, boxes)
            class_loss = self.class_loss_fn(pred_labels, labels)
            
            # the total loss
            loss = boxes_loss + class_loss
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # add to epoch loss
            epoch_box_loss += boxes_loss.item()
            epoch_class_loss += class_loss.item()
            
            # compute how many predictions were right
            if pred_labels.shape[1] == 1:  # binary
                pred_labels = (torch.sigmoid(pred_labels) > 0.5).float()  # Convert logits to binary (0 or 1)
            else:  # Multi-class classification
                predicted_class = pred_labels.argmax(1)  # Take the highest probability cla
            # count how many correct predictions were there
            trainCorrect += (predicted_class == labels).float().sum().item()
            
            # Log losses to TensorBoard
            #TODO finish tensorboard
            self.writer.add_scalar("Loss/BoundingBox", boxes_loss.item(), epoch_idx * len(self.train_dataloader) + batch_idx)
            self.writer.add_scalar("Loss/Classification", class_loss.item(), epoch_idx * len(self.train_dataloader) + batch_idx)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_bbox_loss, total_class_loss = 0, 0
        
        with torch.no_grad():
            #TODO finish
            for images, targets in self.val_dataloader:
                images = [img.to(device) for img in images]
                bboxes = torch.stack([t["bbox"].to(device) for t in targets])
                labels = torch.stack([t["label"].to(device) for t in targets])
                
                images = torch.stack(images)
                
                pred_bboxes, pred_class = self.model(images)
                
                bbox_loss = self.bbox_loss_fn(pred_bboxes, bboxes)
                class_loss = self.class_loss_fn(pred_class, labels)
                
                total_bbox_loss += bbox_loss.item()
                total_class_loss += class_loss.item()
        
        avg_bbox_loss = total_bbox_loss / len(self.val_dataloader)
        avg_class_loss = total_class_loss / len(self.val_dataloader)
        
        print(f"Validation Loss: BBox {avg_bbox_loss:.4f}, Class {avg_class_loss:.4f}")
        
        # Log validation losses to TensorBoard
        self.writer.add_scalar("Val_Loss/BoundingBox", avg_bbox_loss, epoch)
        self.writer.add_scalar("Val_Loss/Classification", avg_class_loss, epoch)


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_bbox_loss, total_class_loss = 0, 0
            
            #TODO finish this, add also tensoroard, add lr scheudler, add auto stopping 
            
            print(f"Train Loss: BBox {total_bbox_loss/len(self.train_dataloader):.4f}, Class {total_class_loss/len(self.train_dataloader):.4f}")
            
            # Validation phase
            self.validate(epoch)
        
        self.writer.close()