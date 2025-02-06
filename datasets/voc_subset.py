# import torch
from torchvision.datasets import VOCDetection
from config import config

# define a class that is the subset of the VOC dataset, with the selected class

class VOCSubset(VOCDetection):
    def __init__(self, indices_list, selected_class, single_object = True, transform=None, target_transform=None):
        ''' Inputs:
                - indices_list: list of indices with the applicable class
                - selected_class: name of selected class in VOC dataset
                - single_object: bool, single object in frame (if no, multiple bboxes per frame returned)
                - transform: the image transformation
                - target_transform: the target (bbox) transformation
                Defeault is download = False, assuming the dataset has been downloaded already.
        '''
        # init the dataset
        super().__init__(root=config.dataset_path,
                        year="2012",
                        image_set="trainval",
                        download=False,
                        transform=transform,
                        target_transform=target_transform)
        
        # store indices
        assert(len(indices_list) > 1)
        self.selected_indices = indices_list
        
        # store params
        self.selected_class = selected_class 
        self.single_object = single_object
        self.root = config.dataset_path

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # get the real image index from the selected indices
        image_idx = self.selected_indices[idx]
        
        # fetch the image using the saved index
        image, target = super().__getitem__(image_idx)  
        
        # the target should only contain annotations for the selected class
        target = self.filter_annotations_for_class(target, self.selected_class)
        return image, target
    
    def filter_annotations_for_class(self, target, selected_class):
        # filter the target to include only annotations from the selected class
        annotations = target['annotation']['object']
        filtered_annotations = []
        
        # append only if the object name (class) is the selected class
        for obj in annotations:
            if obj['name'] == selected_class:
                filtered_annotations.append(obj)
            
                # if single object - stop after the first match
                if self.single_object is True:
                    break  
        
        # replace the original annotations
        target['annotation']['object'] = filtered_annotations
        return target
