import torch
import torchvision
import torchvision.transforms.functional as F

def voc_bbox_to_tensor(voc_bndbox):
    """
    convert VOC-style bounding box notation to a PyTorch tensor.
    """
    xmin = int(voc_bndbox['xmin'])
    ymin = int(voc_bndbox['ymin'])
    xmax = int(voc_bndbox['xmax'])
    ymax = int(voc_bndbox['ymax'])
    
    # the order is permutated in this dataset
    return torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

def vox_img_bbox_plot(image, target, labels = None):
    # get the bounding box for the instance using voc_to_tensor
    obj = target['annotation']['object'][0]['bndbox']
    bbox = voc_bbox_to_tensor(obj)
    
    # convert to uint8
    image_uint8 = (image * 255).to(torch.uint8) 
    # draw the bounding boxes on the image
    image_with_boxes = torchvision.utils.draw_bounding_boxes(image_uint8, bbox, fill=False, colors="red", width=3, labels=labels)
    image_with_boxes_PIL = F.to_pil_image(image_with_boxes)
    
    return image_with_boxes_PIL