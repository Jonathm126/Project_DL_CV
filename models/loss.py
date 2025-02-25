import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss, box_convert


class CIoU(nn.Module):
    def __init__(self, format: str, reduction: str = "mean", eps: float = 1e-7):
        """
        Wraps the complete_box_iou_loss function in a PyTorch nn.Module.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.format = format

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Complete IoU Loss between two sets of bounding boxes.

        Args:
            boxes1: Tensor[N, 4] or Tensor[4] - first set of boxes.
            boxes2: Tensor[N, 4] or Tensor[4] - second set of boxes.

        Returns:
            Tensor: Loss tensor with the reduction option applied.
        """
        boxes1 = box_convert(boxes1, self.format, 'xyxy')
        boxes2 = box_convert(boxes2, self.format, 'xyxy')
        return complete_box_iou_loss(boxes1, boxes2, reduction=self.reduction, eps=self.eps)