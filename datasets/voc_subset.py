# import torch
from torchvision.datasets import VOCDetection
from torchvision.ops import box_area
from torchvision.tv_tensors import BoundingBoxes
from torch import ones_like, zeros_like, float32

# my imports
from config import config
import utils.voc_utils as voc_utils

# define a class that is the subset of the VOC dataset, with the selected class
class VOCSubset(VOCDetection):
    def __init__(self, indices_list, selected_class, single_instance = True, transform = None, transforms = None, download = False):
        ''' Inputs:
                - indices_list: list of indices, some of which have the "selected class"
                - selected_class: name of selected class in VOC dataset. if none, returns all classes
                - single_instance: bool, single object in frame (if no, multiple bboxes per frame returned)
                - transform: a transformation function for the image only
                - transforms: a transformation function for BOTH the image and the target
        '''
        # init the dataset
        super().__init__(root=config.dataset_path,
                        year="2012",
                        image_set='trainval',
                        download = download,
                        transform = None,
                        transforms = None)
        
        # store params
        assert(len(indices_list) >= 1)
        self.selected_indices = indices_list
        self.single_instance = single_instance
        self.both_transform = transforms
        
        # store the selected label as a number after conversion
        if isinstance(selected_class, str):
            # If the selected_class is a string, map it to the corresponding label index
            self.selected_label = voc_utils.voc_class_to_idx[selected_class]
        else:
            # assuming the alternative is an int or a tensor...
            self.selected_label = selected_class

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # get the real image index from the selected indices
        image_idx = self.selected_indices[idx]
        # fetch the image using the saved index
        image, target = super().__getitem__(image_idx)  
        # convert the object to torch notation
        torch_target = voc_utils.parse_target_voc_torch(image, target)
        # filter the result according to the single class \ single instance option
        torch_target = self.single_instance_and_filter(torch_target)
        # call the transform after parsing the target
        image, torch_target = self.both_transform(image, torch_target)
        
        return image, torch_target
    
    def single_instance_and_filter(self, torch_target):
        '''filters annotations per frame based on the following rules:
        - If `self.selected_label` and we are in selected label mode, exists in the image, set labels to 1.
        - If `self.selected_label` and we are in selected label mode, does not exist, set all labels to 0.
        - If `self.single_instance` is True, return only the largest bbox.
        '''
        # start with all boxes and labels
        boxes = torch_target['boxes']
        labels = torch_target['labels']
        
        # get boxes data
        boxes_format = boxes.format
        boxes_canvas_size = boxes.canvas_size
        
        # if single object per image:
        if self.selected_label is not None:
            # filter for objects matching the selected label
            mask = (labels == self.selected_label)
            if mask.sum() > 0:
                # If selected object is in the frame, keep only those objects
                boxes = boxes[mask]
                labels = ones_like(labels[mask], dtype = float32)  # Set labels to 1 for selected objects
            else:
                # If the selected object is NOT in the frame, keep all objects but set labels to 0
                labels = zeros_like(labels, dtype = float32)  # Set all labels to 0
        
        # if single instance per image:
        if self.single_instance:
            # compute the boxes area
            areas = box_area(boxes)
            # get the index of the box with largest area
            largest_idx = areas.argmax().item()
            # return the largest box and label
            boxes = boxes[largest_idx:largest_idx + 1]
            labels = labels[largest_idx:largest_idx + 1]
            
        # Re-wrap boxes as BoundingBoxes to maintain type
        boxes = BoundingBoxes(boxes, format=boxes_format, canvas_size=boxes_canvas_size)
        
        return {'boxes': boxes, 'labels': labels}