# voc_subset.py

import torch
from torchvision.datasets import VOCDetection
from torchvision.ops import box_area, box_convert
from torchvision.tv_tensors import BoundingBoxes

# CHANGED: We now import the shared dataset_path from config.py 
from datasets.config import dataset_path
import utils.voc_utils as voc_utils

class VOCSubset(VOCDetection):
    """
    A subset of the VOC dataset that only uses specified indices and optionally 
    filters for a single selected class (single_instance or multi-instance).
    """
    def __init__(self, indices_list, selected_class, transforms=None, 
                 single_instance=True, download=False):
        """
        Args:
            indices_list (list[int]): subset of indices. 
            selected_class (str or int): name of the selected class in VOC, or a numeric label index.
            transforms (callable): a transform function that takes in (PIL Image, target_dict).
            single_instance (bool): whether to keep only one largest bounding box per image.
            download (bool): Whether to download VOC dataset if not found locally.
        """
        # CHANGED: Use dataset_path from config.py 
        super().__init__(
            root=dataset_path,
            year="2012",
            image_set="trainval",
            download=download,
            transform=None,   # Let both_transform handle everything instead
            transforms=None
        )
        
        assert len(indices_list) >= 1, "indices_list must not be empty"
        self.selected_indices = indices_list
        self.single_instance = single_instance
        self.both_transform = transforms
        
        # Convert the selected_class from string to label index (if needed)
        if isinstance(selected_class, str):
            # For example, if selected_class = 'cat'
            self.selected_label = voc_utils.voc_class_to_idx[selected_class]
        else:
            # If it's already an int, or you are passing a numeric label, just store as is
            self.selected_label = selected_class

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        """
        Returns:
            (image_transformed, bboxes_xywh, labels)
            - image_transformed: the (optionally) transformed image (Tensor)
            - bboxes_xywh: bounding boxes in xywh (normalized to [0,1])
            - labels: typically [1] if selected_class present, else [0]
        """
        # Map from local index in the subset to the original VOC index
        image_idx = self.selected_indices[idx]
        
        # Use the parent class to load the image + annotation
        image, target = super().__getitem__(image_idx)
        
        # Convert the VOC-style annotation => a PyTorch-friendly dict with 'bboxes', 'labels', etc.
        # This depends on your voc_utils implementation
        torch_target = voc_utils.parse_target_voc(image.size[0:2], target)
        
        # Apply the user-supplied transform, if any. 
        # Typically "self.both_transform" is a function that expects (PIL.Image, dict).
        image_transformed, transformed_target_dict = self.both_transform(image, torch_target)
        
        # The new bounding boxes are stored in transformed_target_dict['bboxes'] (a TVTensor).
        # We'll normalize them to [0,1] by dividing by the canvas size
        canvas_size = torch.tensor(transformed_target_dict['bboxes'].canvas_size)  # (H, W)
        bboxes_norm = transformed_target_dict['bboxes'] / torch.cat((canvas_size, canvas_size), dim=0)
        
        # Filter to keep only selected_class, etc.
        bboxes_clean, labels_clean = self.single_instance_and_filter(
            bboxes_norm,
            transformed_target_dict['labels']
        )
        
        # Convert from xyxy => xywh
        bboxes_clean_xywh = box_convert(bboxes_clean, 'xyxy', 'xywh')
        
        # Wrap it back in a TVTensor if you like, for convenience
        bboxes_clean_xywh = BoundingBoxes(
            bboxes_clean_xywh, 
            format='xywh', 
            canvas_size=canvas_size, 
            dtype=torch.float32
        )
        
        return image_transformed, bboxes_clean_xywh, labels_clean
    
    def single_instance_and_filter(self, bboxes, labels):
        """
        Filters annotations in the following way:
         - if self.selected_label is not None, keep only those bounding boxes matching selected_label 
           (set label to 1). If none match, label = 0.
         - if self.single_instance=True, pick only the bounding box with the largest area.
        Returns:
            (filtered_bboxes, filtered_labels)
        """
        # If we have a specific class to keep
        if self.selected_label is not None:
            mask = (labels == self.selected_label)
            if mask.sum() > 0:
                # The selected object is in the frame => keep those objects, set label = 1
                bboxes = bboxes[mask]
                labels = torch.ones_like(labels[mask], dtype=torch.float16)
            else:
                # The selected class isn't in the frame => label = 0, (or you could keep an empty box)
                labels = torch.zeros_like(labels, dtype=torch.float16)
                # Potentially you could also zero out bboxes, but typically returning an empty box
        
        # If single instance, keep largest bounding box only
        if self.single_instance and len(bboxes) > 0:
            areas = box_area(bboxes)
            largest_idx = areas.argmax().item()
            bboxes = bboxes[largest_idx:largest_idx + 1]
            labels = labels[largest_idx:largest_idx + 1]
        
        return bboxes, labels
