# import torch
from torchvision.datasets import VOCDetection
from torchvision.ops import box_area

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
        
        # if the target should only contain a single object (instance) per frame
        torch_target = self.single_instance_and_filter(torch_target)
        
        # call the transform after parsing the target
        image, torch_target = self.both_transform(image, torch_target)
        
        # converts to binary labels, true if the label is the model selected label
        # TODO this was moved to 
        # torch_target['labels'] = (torch_target['labels'] == self.selected_label).float()
        
        return image, torch_target
    
    # def single_instance_and_filter(self, torch_target):
    # TODO Obsolete
    #     '''Filter the annotations to include only one object for image. If this objet is the selected class, it will be presented.
    #         Input: target in torch format.'''
    #     # TODO this gives priority to the selected target and also filters. in multiple instances we will need to change
        
    #     # if the selected class is in the image
    #     if self.selected_label in torch_target['labels']:
    #         # get the index of the first insacnce of the selected label
    #         selected_idx = (torch_target['labels'] == self.selected_label).nonzero(as_tuple=False)[0].item()
            
    #         # filter the target dict
    #         torch_target['boxes'].data = torch_target['boxes'].data[selected_idx:selected_idx+1]
    #         torch_target['labels'] = torch_target['labels'][selected_idx:selected_idx+1]
        
    #     # if the selected class is not in the image, return first instance of the other label
    #     else:
    #         torch_target['boxes'].data = torch_target['boxes'].data[:1]
    #         torch_target['labels'] = torch_target['labels'][:1]

    #     return torch_target
    
    def single_instance_and_filter(self, torch_target):
        '''filters annotations per frame based on the following rules:
        - If `self.selected_label` and we are in selected label mode, exists in the image, set labels to 1.
        - If `self.selected_label` and we are in selected label mode, does not exist, set all labels to 0.
        - If `self.single_instance` is True, return only the largest bbox.
        '''
        # start with all boxes and labels
        boxes = torch_target['boxes']
        labels = torch_target['labels']
        
        # if single object per image:
        if self.selected_label is not None:
            # filter for objects matching the selected label
            mask = (labels == self.selected_label)
            if mask.sum() > 0:
            # if the selected object is in the frame - return only that
                boxes = boxes[mask]
                labels = labels[mask]
            # else: no object matches the selected label → fall back to all objects
        
        # if single instance per image:
        if self.single_instance:
            # compute the boxes area
            areas = box_area(boxes)
            # get the index of the box with largest area
            largest_idx = areas.argmax().item()
            # return the largest box and label
            boxes = boxes[largest_idx:largest_idx + 1]
            labels = labels[largest_idx:largest_idx + 1]
        
        return {'boxes': boxes, 'labels': labels}