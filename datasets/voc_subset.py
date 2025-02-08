# import torch
from torchvision.datasets import VOCDetection
from config import config
import utils.voc_utils as voc_utils

# define a class that is the subset of the VOC dataset, with the selected class
class VOCSubset(VOCDetection):
    def __init__(self, indices_list, selected_class, single_instance = True, transform = None, target_transform = None):
        ''' Inputs:
                - indices_list: list of indices, some of which have the "selected class"
                - selected_class: name of selected class in VOC dataset
                - single_instance: bool, single object in frame (if no, multiple bboxes per frame returned)
                - transform: the image transformation
                - target_transform: the target (bbox) transformation
        '''
        # init the dataset
        super().__init__(root=config.dataset_path,
                        year="2012",
                        image_set="trainval",
                        download=False,
                        transform=transform,
                        target_transform=target_transform)
        
        # store indices
        assert(len(indices_list) >= 1)
        self.selected_indices = indices_list
        
        # store the selected label as a number after conversion
        if isinstance(selected_class, str):
            # If the selected_class is a string, map it to the corresponding label index
            self.selected_label = voc_utils.voc_class_to_idx[selected_class]
        else:
            # assuming the alternative is an int or a tensor...
            self.selected_label = selected_class

        self.single_instance = single_instance

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # get the real image index from the selected indices
        image_idx = self.selected_indices[idx]
        # fetch the image using the saved index
        image, target = super().__getitem__(image_idx)  
        # convert the object to torch notation
        torch_target = voc_utils.parse_target_voc_torch(target)
        # if the target should only contain a single object (instance) per frame
        if self.single_instance:
            torch_target = self.single_instance_and_filter(torch_target)
        
        return image, torch_target
    
    def single_instance_and_filter(self, torch_target):
        '''Filter the annotations to include only one object for image. If this objet is the selected class, it will be presented.
            Input: target in torch format.'''
        # TODO this gives priority to the selected target and also filters. in multiple instances we will need to change
        filtered_torch_target = {}
        
        # if the selected class is in the image
        if self.selected_label in torch_target['labels']:
            # get the index of the first insacnce of the selected label
            selected_idx = (torch_target['labels'] == self.selected_label).nonzero(as_tuple=False)[0].item()
            # filter the target dict
            filtered_torch_target['boxes'] = torch_target['boxes'][selected_idx:selected_idx+1]
            filtered_torch_target['labels'] = torch_target['labels'][selected_idx:selected_idx+1]
        
        # if the selected class is not in the image, return first instance of the other label
        else:
            filtered_torch_target['boxes'] = torch_target['boxes'][:1]
            filtered_torch_target['labels'] = torch_target['labels'][:1]

        return filtered_torch_target
