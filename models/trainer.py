# torch
import torch
import torch.nn as nn
import torch.optim as optim

# visualizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, device, model, train_dataloader, val_dataloader, losses, optimizer, max_epochs=10):
        # init
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = max_epochs
        self.optimizer = optimizer
        
        # define loss functions
        self.bbox_loss_fn = losses[0] # for the bounding box use the first loss type
        self.class_loss_fn = losses[1] # for the classifier use the second loss type
        
        # TensorBoard writer
        self.writer = SummaryWriter()

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_bbox_loss, total_class_loss = 0, 0
            
            for batch_idx, (images, targets) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")):
                images = [img.to(self.device) for img in images]
                bboxes = torch.stack([t["bbox"].to(self.device) for t in targets])  # (batch, 4)
                labels = torch.stack([t["label"].to(self.device) for t in targets])  # (batch, 1)
                
                images = torch.stack(images)  # Convert list to tensor (batch, C, H, W)
                self.optimizer.zero_grad()
                
                pred_bboxes, pred_class = self.model(images)
                
                bbox_loss = self.bbox_loss_fn(pred_bboxes, bboxes)
                class_loss = self.class_loss_fn(pred_class, labels)
                
                loss = bbox_loss + class_loss
                loss.backward()
                self.optimizer.step()
                
                total_bbox_loss += bbox_loss.item()
                total_class_loss += class_loss.item()
                
                # Log losses to TensorBoard
                self.writer.add_scalar("Loss/BoundingBox", bbox_loss.item(), epoch * len(self.train_dataloader) + batch_idx)
                self.writer.add_scalar("Loss/Classification", class_loss.item(), epoch * len(self.train_dataloader) + batch_idx)
            
            print(f"Train Loss: BBox {total_bbox_loss/len(self.train_dataloader):.4f}, Class {total_class_loss/len(self.train_dataloader):.4f}")
            
            # Validation phase
            self.validate(epoch)
        
        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_bbox_loss, total_class_loss = 0, 0
        
        with torch.no_grad():
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
