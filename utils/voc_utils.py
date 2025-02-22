import torch
from torchvision import tv_tensors
from torchvision.ops import box_convert

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

def parse_target_voc_torch(image, target):
    '''Parse the VOC target from VOC notation to torch notation.'''
    torch_target, boxes, labels = {}, [], []
    
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
    
    # convert to torchvision tensor of type boundingbox
    boxes_tensor = torch.stack(boxes)
    
    # normalize the tensors and convert to xywh
    img_w, img_h = image.size
    # TODO cancelled xywh
    # boxes_tensor_xywh = box_convert(boxes_tensor, in_fmt = 'xyxy', out_fmt = 'xywh')
    # boxes_tensor_xywh = boxes_tensor
    
    # convert to dict with Tvtensors
    torch_target['boxes'] = tv_tensors.BoundingBoxes(boxes_tensor, format='xyxy',  # cancelled xywh
                                                canvas_size=(img_h, img_w), dtype=torch.float32)
    torch_target['labels'] = torch.tensor(labels, dtype=torch.int64)  # shape: (N,1)
    
    return torch_target
