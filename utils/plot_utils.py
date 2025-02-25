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

# def voc_img_bbox_plot(image, boxes1, labels1, boxes2 = None, labels2 = None):
#     '''Helper function to plot bounding boxes on a SINGLE image, given labels and boxes.
#         Input:
#         - image1 - torch float32
#         - boxes, labels - torch format target structure 
#         - image2, boxes2, labels2 - optinal - similar
#         Output: Tensor image
#     '''
#     # convert iamge to uint8
#     image_uint8 = (image * 255).to(torch.uint8) 
#     h, w = image.shape[-2:]
#     # unnormalize bbox and transform to xyxy
#     boxes1 = boxes1 * torch.tensor([h, w, h, w], dtype=torch.float32, device = boxes1.device)
#     boxes1 = box_convert(boxes1, 'xywh', 'xyxy')
#     # convert labels to a list of strings only if they are a tensor
#     if isinstance(labels1, torch.Tensor):
#         labels1 = [str(label.item()) for label in labels1] 
#     # process target1 (red boxes)
#     try: # handle case of illeagl bbox
#         image_with_boxes = torchvision.utils.draw_bounding_boxes(image_uint8, boxes1, fill=False, colors="red", width=3, 
#                                                                 labels=labels1, font_size=25, font = 'verdana.ttf',)
#     except Exception:
#         image_with_boxes = image_uint8
#     # handle target2 if present
#     if boxes2 is not None:
#         # unnormalize bbox and transform to xyxy
#         boxes2 = boxes2 * torch.tensor([h, w, h, w], dtype=torch.float32, device=boxes2.device)
#         boxes1 = box_convert(boxes1, 'xywh', 'xyxy')
#         # process labels
#         if isinstance(labels2, torch.Tensor):
#             labels2 = [str(label.item()) for label in labels2] 
#         try: # handle case of illeagl bbox
#             image_with_boxes = torchvision.utils.draw_bounding_boxes(image_with_boxes, boxes2, fill=False, colors="blue", width=3, 
#                                                                     labels=labels2, font_size=25, font = 'verdana.ttf')
#         except Exception:
#             pass
        
#     return image_with_boxes

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
