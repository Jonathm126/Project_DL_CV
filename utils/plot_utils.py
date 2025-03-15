import random
import matplotlib.pyplot as plt

import torch
import torchvision 
import torchvision.transforms.functional as F
from torchvision.ops import box_convert

def unnormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse a normalization transform by applying the inverse mean/std.
    """
    inv_mean = [-m/s for m, s in zip(mean, std)]
    inv_std = [1.0/s for s in std]
    return F.normalize(image, mean=inv_mean, std=inv_std)

def plot_images_from_voc_dataset(dataset, num_images=8, title="Dataset Images"):
    """
    Plots random images from a given dataset.
    Inputs:
        dataset: PyTorch Dataset object returning (image, bboxes, labels).
        num_images (int): Number of images to plot.
        title (str): Title for the plot.
    """
    random_indices = random.sample(range(len(dataset)), num_images)
    
    fig, axes = plt.subplots(num_images // 4, 4, figsize=(12, 4))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        # Get image and target from the dataset
        image, bboxes, labels = dataset[idx]

        # Convert numeric labels into strings like 'Cat' / 'Else' for display
        labels = ['Cat' if label == 1 else 'Else' for label in labels]

        # Unnormalize the image
        image_un_norm = unnormalize(image)

        # Draw bounding boxes
        image_with_boxes = voc_img_bbox_plot(image_un_norm, bboxes, labels)

        # Convert to PIL for plotting in matplotlib
        image_with_boxes_pil = F.to_pil_image(image_with_boxes)
        
        axes[i].imshow(image_with_boxes_pil)
        axes[i].axis("off")
        axes[i].set_title(f"Image {idx}, label: {labels[0]}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_single_box(image, boxes, labels, color):
    """
    Helper to draw bounding boxes on a single image tensor.

    Args:
        image (Tensor): Shape [3, H, W], float or uint8.
        boxes (Tensor): Shape [N, 4], in xywh format, normalized [0..1].
        labels (List[str] or Tensor): Per-box labels.
        color (str): e.g. "red" or "blue".
    """
    if boxes is None or len(boxes) == 0:
        return image

    # Dimensions
    h, w = image.shape[-2:]

    # Scale from [0..1] to pixel coords, then convert xywh => xyxy
    boxes_pixel = boxes * torch.tensor([h, w, h, w], dtype=torch.float32, device=boxes.device)
    boxes_xyxy = box_convert(boxes_pixel, 'xywh', 'xyxy')
    
    # Convert labels if it's a tensor
    if isinstance(labels, torch.Tensor):
        labels = [str(l.item()) for l in labels]

    # IMPORTANT: move image to CPU (and ensure uint8) before draw_bounding_boxes
    image_cpu = image.cpu()
    boxes_xyxy_cpu = boxes_xyxy.cpu()

    try:
        image_annotated = torchvision.utils.draw_bounding_boxes(
            image_cpu,
            boxes_xyxy_cpu,
            colors=color,
            width=3,
            labels=labels,
            font_size=25
        )
    except Exception:
        # If any error arises from invalid boxes, just skip
        return image_cpu

    return image_annotated

def voc_img_bbox_plot(image, boxes1, labels1, boxes2=None, labels2=None):
    """
    Plots up to two sets of bounding boxes (GT + predicted) on a single image.

    Args:
        image (Tensor): Float32 image [3,H,W], typically in [0..1].
        boxes1 (Tensor): shape [N,4], xywh in [0..1].
        labels1 (List[str] or Tensor): label texts for boxes1.
        boxes2 (Tensor): optional second set of boxes.
        labels2 (List[str] or Tensor): optional second set of labels.

    Returns:
        Tensor: The final annotated image in uint8 on CPU.
    """
    # Convert to uint8 for drawing
    image_uint8 = (image * 255).to(torch.uint8)

    # Draw first set (in red)
    image_with_boxes = plot_single_box(image_uint8, boxes1, labels1, color="red")

    # Draw second set (in blue), if provided
    if boxes2 is not None and len(boxes2) > 0:
        image_with_boxes = plot_single_box(image_with_boxes, boxes2, labels2, color="blue")
    
    return image_with_boxes

def inverse_transform_bbox(bboxes: torch.Tensor, W: float, H: float) -> torch.Tensor:
    """
    Converts bounding boxes from normalized crop coordinates (xywh) to
    original image coordinates (xyxy), given an original W,H.

    Args:
        bboxes (Tensor): shape [N,4], in normalized xywh in [0..1].
        W (float): original image width.
        H (float): original image height.

    Returns:
        Tensor of shape [N,4] in xyxy format in the original image coordinates.
    """
    # Suppose your resizing short side to 232, then center-cropping to 224
    s = 232.0 / min(W, H)
    resized_W = W * s
    resized_H = H * s

    # Offsets for a 224x224 center crop
    offset_x = (resized_W - 224) / 2.0
    offset_y = (resized_H - 224) / 2.0

    # Scale from [0..1] to [0..224], then shift back to resized coords
    x_crop = bboxes[..., 0] * 224.0
    y_crop = bboxes[..., 1] * 224.0
    w_crop = bboxes[..., 2] * 224.0
    h_crop = bboxes[..., 3] * 224.0

    x_resized = offset_x + x_crop
    y_resized = offset_y + y_crop

    # Map from resized coords back to original
    x_original = x_resized / s
    y_original = y_resized / s
    w_original = w_crop / s
    h_original = h_crop / s

    x_max = x_original + w_original
    y_max = y_original + h_original

    return torch.stack([x_original, y_original, x_max, y_max], dim=-1)
