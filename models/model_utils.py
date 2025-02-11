import torch
import matplotlib.pyplot as plt
import random
from utils.plot_utils import voc_img_bbox_plot

def evaluate_model(model, dataset, num_images=8, title="Model Predictions", device="cuda"):
    """
    evaluates the model by plotting its predictions on random dataset samples.
    """
    model.eval()  
    random_indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(num_images // 4, 4, figsize=(15, 8))
    axes = axes.flatten()

    with torch.no_grad():  
        for i, idx in enumerate(random_indices):
            # get image and target
            image, _ = dataset[idx]  
            image = image.to(device).unsqueeze(0)  # add batch dimension
            
            # run model inference
            pred_bboxes, pred_labels = model(image)
            
            # utilize sigmoid
            labels = (torch.sigmoid(pred_labels) > 0.5).long()
            
            # convert predictions to torch format (matching dataset format)
            pred_target = {
                "boxes": pred_bboxes.cpu(),  # convert to (num_boxes, 4)
                "labels": labels.cpu()  # convert logits to binary labels
            }

            # convert image & draw predicted bounding boxes
            image_with_boxes_PIL = voc_img_bbox_plot(image.squeeze(0).cpu(), pred_target)

            # Plot the image
            axes[i].imshow(image_with_boxes_PIL)
            axes[i].axis("off")
            axes[i].set_title(f"Pred Image {idx}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
