import torch
import matplotlib.pyplot as plt
import random
from utils.plot_utils import voc_img_bbox_plot, unnormalize

def evaluate_model(model, dataset, num_images=8, device="cuda"):
    # TODO replace with tensorboard
    """
    evaluates the model by plotting its predictions on random dataset samples.
    """
    model.eval()  
    random_indices = random.sample(range(len(dataset)), num_images)
    
    num_rows = (num_images + 3) // 4  
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    with torch.no_grad():  
        for i, idx in enumerate(random_indices):
            # get image and target
            image, target = dataset[idx]  
            image = image.to(device).unsqueeze(0)  # add batch dimension
            
            # un-normalize the image
            mean, std = model.backbone_transforms().mean, model.backbone_transforms().std
            image = unnormalize(image, mean, std)
            
            # run model inference
            pred_bboxes, pred_labels = model(image)
            target_label = target['labels']
            
            # utilize sigmoid
            labels = (torch.sigmoid(pred_labels) > 0.5).long()
            
            # handle labels
            pred_label_str = ["cat" if label.item() == 1 else "not cat" for label in labels]
            target_label_str = ["cat" if label.item() == 1 else "not cat" for label in target_label]
            
            # convert predictions to torch format (matching dataset format)
            pred_target = {
                "boxes": pred_bboxes.cpu(),  # convert to (num_boxes, 4)
                "labels": labels.cpu()  # convert logits to binary labels
            }

            # convert image & draw predicted bounding boxes
            image_with_boxes_PIL = voc_img_bbox_plot(image.squeeze(0).cpu(), pred_target, target)

            # Plot the image
            axes[i].imshow(image_with_boxes_PIL)
            axes[i].axis("off")
            axes[i].set_title(f"Pred: {pred_label_str}, Target:{target_label_str}")

    plt.tight_layout()
    plt.show()
