import torch
import torchvision
import torchvision.transforms.functional as F

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
    idx = [class_names_list[label - 1] for label in labels]
    
    return idx

# helper for plotting
def voc_img_bbox_plot(image, target):
    '''Helper function to plot bounding boxes.
        Input:
        - image - torch float32
        - target - in torch notation
        - labels (optional): list of names'''
    
    # get the bounding box for the instance using voc_to_tensor
    boxes = target['boxes']
    # convert the labels to string from number
    labels = voc_idx_to_class(target['labels'])
    # convert iamge to uint8
    image_uint8 = (image * 255).to(torch.uint8) 
    # draw the bounding boxes on the image
    image_with_boxes = torchvision.utils.draw_bounding_boxes(image_uint8, boxes, fill=False, colors="red", width=3, labels=labels)
    image_with_boxes_PIL = F.to_pil_image(image_with_boxes)
    
    return image_with_boxes_PIL

# helper functions for parsing voc to torch
def voc_bbox_to_tensor(voc_bndbox):
    '''Convert VOC-style bounding box notation to a PyTorch tensor.'''
    xmin = int(voc_bndbox['xmin'])
    ymin = int(voc_bndbox['ymin'])
    xmax = int(voc_bndbox['xmax'])
    ymax = int(voc_bndbox['ymax'])
    
    # return in torch notation order
    return torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

def parse_target_voc_torch(target):
    '''Parse the VOC target from VOC notation to torch notation.'''
    torchTarget, boxes, labels = {}, [], []
    
    # loop for each detected object:
    for obj in target['annotation']['object']:
        # get bouding box
        boxes.append(voc_bbox_to_tensor(obj['bndbox'])) # append the bounding box
        labels.append(voc_class_to_idx[obj['name']])    # append the label (name)
        
    torchTarget['boxes'] = torch.stack(boxes)  # shape: (N, 4)
    torchTarget['labels'] = torch.tensor(labels, dtype=torch.int64)  # shape: (N,)
    
    return torchTarget