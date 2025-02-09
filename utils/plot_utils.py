import random
import matplotlib.pyplot as plt

# torch
import torch
import torchvision 
import torchvision.transforms.functional as F

# my imports
from utils.voc_utils import voc_idx_to_class

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
    
    fig, axes = plt.subplots(num_images // 4, 4, figsize=(15, 8))
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
def voc_img_bbox_plot(image, target):
    '''Helper function to plot bounding boxes.
        Input:
        - image - torch float32
        - target - in torch notation
        - labels (optional): list of names
    '''
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
