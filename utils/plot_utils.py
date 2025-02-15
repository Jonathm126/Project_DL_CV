import random
import matplotlib.pyplot as plt

# torch
import torch
import torchvision 
import torchvision.transforms.functional as F

# my imports
from utils import voc_utils

# un-normalize a normalized image
def unnormalize(image, mean, std):
    """
    Returns a transform that reverses normalization.
    """
    inv_mean = [-m/s for m, s in zip(mean, std)]
    inv_std = [1/s for s in std]
    return F.normalize(image, mean=inv_mean, std=inv_std)

# function to visualize images from a dataset
def plot_images_from_voc_dataset(dataset, num_images=8, title="Dataset Images"):
    """
    Plots random images from a given dataset.
    Inputs:
        dataset: PyTorch Dataset object.
        num_images (int): Number of images to plot.
        title (str): Title for the plot.
    """
    random_indices = random.sample(range(len(dataset)), num_images)
    
    fig, axes = plt.subplots(num_images // 4, 4, figsize=(12, 4))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        # Get image and target
        image, torch_target = dataset[idx]
        
        # Convert image and draw bounding boxes
        image_with_boxes_PIL = voc_img_bbox_plot(image, torch_target)
        
        # Plot the image
        axes[i].imshow(image_with_boxes_PIL)
        axes[i].axis("off")
        axes[i].set_title(f"Image {idx}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
# helper for transforming voc bbox to PIL
def voc_img_bbox_plot(image, boxes1, labels1, boxes2 = None, labels2 = None):
    '''Helper function to plot bounding boxes on an image, given labels and boxes.
        Input:
        - image1 - torch float32
        - boxes, labels - torch format target structure
        - image2, boxes2, labels2 - optinal - similar
        Output: tensor.
    '''
    # convert iamge to uint8
    image_uint8 = (image * 255).to(torch.uint8) 
    
    # Process target1 (Red boxes)
    image_with_boxes = torchvision.utils.draw_bounding_boxes(image_uint8, 
                                                            boxes1, fill=False, colors="red", width=3, labels=labels1)
    
    # handle target2 if present
    if boxes2:
        image_with_boxes = torchvision.utils.draw_bounding_boxes(image_with_boxes, 
                                                                boxes2, fill=False, colors="blue", width=3, labels=labels2)
    return image_with_boxes


def scsi_images_bbox_grid(img, target, pred, backbone_transforms = None):
    """
    Helper function to plot bounding boxes and labels for the single class, single instance case. 
    Handles conversion of the label, etc.
    Inputs:
    - img: original image
    - target: ground truth data
    - pred: model's prediction
    - backbone transforms: for un-normalization
    """
    pred_bboxes, pred_labels = pred
    
    # sigmoid for binary classification
    pred_labels = (torch.sigmoid(pred_labels) > 0.5).long()
    
    # handle labels - convert from number to string
    pred_label_str = voc_utils.voc_idx_to_class(pred_labels.squeeze(0).tolist())
    target_label_str = voc_utils.voc_idx_to_class(target["labels"].squeeze(0).tolist())
    
    # un-normalize image if data is supplied
    if backbone_transforms:
        mean, std = backbone_transforms().mean, backbone_transforms().std
        img = unnormalize(img, mean, std) 
        
    # get bbox
    return voc_img_bbox_plot(img.squeeze(0).cpu(), 
                    target["boxes"].cpu(), target_label_str,
                    pred_bboxes.cpu(), pred_label_str)