import random
import matplotlib.pyplot as plt

# torch
import torch
import torchvision 
import torchvision.transforms.functional as F
from torchvision.ops import box_convert

# un-normalize a normalized image
def unnormalize(image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
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
    
    _, axes = plt.subplots(num_images // 4, 4, figsize=(12, 4))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        # Get image and target
        image, bboxes, labels = dataset[idx]
        # convert labels
        labels = ['Cat' if label == 1 else 'Else' for label in labels]
        # unnormalize image and draw bounding boxes
        image_un_norm = unnormalize(image)
        image_with_boxes = voc_img_bbox_plot(image_un_norm, bboxes, labels)
        image_with_boxes_PIL = F.to_pil_image(image_with_boxes)
        # Plot the image
        axes[i].imshow(image_with_boxes_PIL)
        axes[i].axis("off")
        axes[i].set_title(f"Image {idx}, label: {labels[0]}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_single_box(image, boxes, labels, color):
    """Helper function to process and plot bounding boxes and labels for a single image.
    Input:
    - image
    - boxes: (N, 4), 'xywh'
    - labels: (N,1)
    - color 'str'
    Output: Image with annotation
    """
    h, w = image.shape[-2:]
    
    if boxes is None:
        return image
    
    # Unnormalize bbox and transform to xyxy
    boxes = boxes * torch.tensor([h, w, h, w], dtype=torch.float32, device=boxes.device)
    boxes = box_convert(boxes, 'xywh', 'xyxy')
    
    # Convert labels to a list of strings if they are tensors
    if isinstance(labels, torch.Tensor):
        labels = [str(label.item()) for label in labels]
    
    try:  # Handle cases of illegal bbox
        image = torchvision.utils.draw_bounding_boxes(image, boxes, fill=False, colors=color, width=3, 
                                                    labels=labels, font_size=25, font='verdana.ttf')
    except Exception:
        pass
    
    return image

def voc_img_bbox_plot(image, boxes1, labels1, boxes2=None, labels2=None):
    '''Helper function to plot bounding boxes on a SINGLE image, given labels and boxes.
        Input:
        - image - torch float32
        - boxes1, labels1 - torch format target structure 
        - boxes2, labels2 - optional - similar
        Output: Tensor image
    '''
    # Convert image to uint8
    image_uint8 = (image * 255).to(torch.uint8)
    
    # Process first set of bounding boxes (red)
    image_with_boxes = plot_single_box(image_uint8, boxes1, labels1, "red")
    
    # Process second set of bounding boxes (blue), if present
    if boxes2 is not None:
        image_with_boxes = plot_single_box(image_with_boxes, boxes2, labels2, "blue")
    
    return image_with_boxes


def inverse_transform_bbox(bboxes: torch.Tensor, original_size, resize_size, crop_size):
    """
    Inverts the Resize and CenterCrop transforms on a tensor of bounding boxes with shape [B, N, 4].

    Args:
        bboxes (torch.Tensor): Tensor of shape [B, N, 4] with bounding boxes in cropped coordinates,
                               where each bbox is [x_min, y_min, x_max, y_max].
        original_size (tuple): Original image dimensions as (orig_h, orig_w).
        resize_size (tuple): Size used in T.Resize, e.g., (R_h, R_w).
        crop_size (int or tuple): Size used in T.CenterCrop. If int, assumes a square crop.

    Returns:
        torch.Tensor: Bounding boxes in original image coordinates with shape [B, N, 4].
    """
    # Ensure crop_size is a tuple (crop_h, crop_w)
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size

    R_h, R_w = resize_size
    orig_h, orig_w = original_size

    # Compute center crop offsets.
    offset_x = (R_w - crop_w) // 2
    offset_y = (R_h - crop_h) // 2

    # Create an offsets tensor with shape [1, 1, 4] for broadcasting.
    offsets = torch.tensor([offset_x, offset_y, offset_x, offset_y],
                             dtype=bboxes.dtype, device=bboxes.device).view(1, 1, 4)
    
    # Reverse the cropping by adding the offsets.
    bboxes_resized = bboxes + offsets

    # Compute scaling factors to reverse the resize.
    scale_x = orig_w / R_w
    scale_y = orig_h / R_h

    # Create a scales tensor with shape [1, 1, 4] for broadcasting.
    scales = torch.tensor([scale_x, scale_y, scale_x, scale_y],
                          dtype=bboxes.dtype, device=bboxes.device).view(1, 1, 4)
    
    # Apply scaling to map the bounding boxes back to the original image coordinates.
    bboxes_original = bboxes_resized * scales

    # Optionally, if you want integer coordinates, you can round them:
    bboxes_original = bboxes_original.round()
    # To convert to integer tensor, uncomment the following line:
    # bboxes_original = bboxes_original.to(torch.int)

    return bboxes_original
