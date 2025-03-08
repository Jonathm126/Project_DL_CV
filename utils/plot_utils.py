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


def inverse_transform_bbox(bboxes: torch.Tensor, W: float, H: float) -> torch.Tensor:
    """
    Converts bounding boxes from normalized crop coordinates (xywh) to original image coordinates (xyxy).

    Args:
        bboxes (torch.Tensor): Tensor of shape [N, 4] containing normalized bounding boxes in (x, y, w, h) format.
                                Values are in range [0, 1] relative to the 224x224 crop.
        W (float): Original image width.
        H (float): Original image height.

    Returns:
        torch.Tensor: Tensor of shape [N, 4] with bounding boxes in the original image coordinate system in (xyxy) format.
    """
    # Compute the scale factor used during resize.
    # The shorter side is resized to 232.
    s = 232.0 / min(W, H)
    
    # Dimensions of the resized image.
    resized_W = W * s
    resized_H = H * s
    
    # Compute the center crop offsets for a 224x224 crop.
    offset_x = (resized_W - 224) / 2.0
    offset_y = (resized_H - 224) / 2.0

    # Convert normalized crop coordinates to pixel coordinates in the crop (shape: [N, 4]).
    # Multiply by 224 since the crop size is 224x224.
    x_crop = bboxes[..., 0] * 224.0
    y_crop = bboxes[..., 1] * 224.0
    w_crop = bboxes[..., 2] * 224.0
    h_crop = bboxes[..., 3] * 224.0

    # Map from crop coordinates to resized image coordinates by adding the crop offset.
    x_resized = offset_x + x_crop
    y_resized = offset_y + y_crop

    # Scale back from the resized image to the original image dimensions.
    x_original = x_resized / s
    y_original = y_resized / s
    w_original = w_crop / s
    h_original = h_crop / s
    
    # Convert from (x, y, w, h) to (x_min, y_min, x_max, y_max)
    x_max = x_original + w_original
    y_max = y_original + h_original

    boxes_xyxy = torch.stack([x_original, y_original, x_max, y_max], dim=-1)
        
    return boxes_xyxy


