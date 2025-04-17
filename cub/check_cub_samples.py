#!/usr/bin/env python3
"""
check_cub_samples.py

A script to quickly visualize a few samples from the processed CUB dataset,
plotting bounding boxes and labels to confirm data integrity.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from cub_datamodule import CUBDataModule  # or wherever your data module is


def plot_sample(ax, image_tensor, class_id, bbox, class_names=None):
    """
    Plots a single sample on the given axes:
      - image_tensor: shape [3, H, W] or [H, W, 3] after convert
      - class_id: integer
      - bbox: [x_min, y_min, x_max, y_max]
      - class_names: optional list or dict for mapping class_id -> name
    """
    # Convert CHW -> HWC for plt
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Clip or scale if needed for display
    img_np = img_np.clip(0, 1)

    ax.imshow(img_np)

    # Draw bounding box
    x_min, y_min, x_max, y_max = bbox.tolist()
    rect_w = x_max - x_min
    rect_h = y_max - y_min
    rect = patches.Rectangle(
        (x_min, y_min), rect_w, rect_h, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # Label
    if class_names is not None and class_id < len(class_names):
        title_str = f"{class_names[class_id]} (cls {class_id})"
    else:
        title_str = f"cls {class_id}"
    ax.set_title(title_str, fontsize=8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a few samples from CUB DataModule."
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed_cub",
        type=str,
        help="Where train.csv, val.csv, images/ are located",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for loading samples"
    )
    parser.add_argument(
        "--num_workers", default=0, type=int, help="Number of workers for DataLoader"
    )
    args = parser.parse_args()

    # Set up data module (simple transforms for display)
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.ToTensor(),  # 0..1 scale
        ]
    )
    dm = CUBDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
    )
    dm.setup("fit")  # create train/val datasets

    # For a real class_names list, you'd parse from classes.txt or store in a file.
    # This is a dummy example for 200 classes:
    class_names = [f"cls_{i}" for i in range(200)]

    # Grab one batch from the train_dataloader
    train_loader = dm.train_dataloader()
    images, (labels, bboxes) = next(iter(train_loader))

    # Plot them in a grid
    fig, axs = plt.subplots(1, min(args.batch_size, 4), figsize=(12, 4))

    # Ensure axs is a 1D list/array of Axes
    import numpy as np

    if isinstance(axs, np.ndarray):
        axs = axs.ravel()
    else:
        axs = [axs]

    for i, ax in enumerate(axs):
        if i >= images.size(0):
            break
        img = images[i]
        cls_id = labels[i].item()
        box = bboxes[i]
        plot_sample(ax, img, cls_id, box, class_names=class_names)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
