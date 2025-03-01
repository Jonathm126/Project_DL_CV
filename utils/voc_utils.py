import torch
from torchvision.tv_tensors import BoundingBoxes
import numpy as np

# define the voc class to numerical index dict
voc_class_to_idx = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

def voc_idx_to_class(labels):
    ''' Does the reverse of voc_class_to_idx.'''
    class_names_list = list(voc_class_to_idx.keys())
    # ensure `labels` is a list (convert single integer to list)
    if isinstance(labels, int):  
        labels = [labels]  
    elif isinstance(labels, torch.Tensor):  
        labels = labels.tolist() 
    idx = [class_names_list[int(label) - 1] for label in labels]
    return idx

def parse_target_voc(img_shape, target):
    '''Parse the VOC target from VOC dataset, return torch and np'''
    boxes, labels = [], []
    w, h = img_shape
    
    # loop for each detected object:
    for obj in target['annotation']['object']:
        # get bouding box
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        
        # append the bounding box
        boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)) 
        # append the label (name)
        labels.append(voc_class_to_idx[obj['name']])    
    
    # convert to torch
    bboxes = BoundingBoxes(torch.stack(boxes), format = 'xyxy', canvas_size = (h, w), dtype = torch.float32)
    labels = torch.tensor(labels, dtype = torch.float16)
    
    return {'bboxes': bboxes, 'labels': labels}
