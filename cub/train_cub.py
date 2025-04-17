#!/usr/bin/env python3
"""
train_cub.py

A PyTorch Lightning training script for the CUB dataset, with two heads:
  1) Classification head (multi-class classification)
  2) Regression head (bounding box)

Allows command-line arguments to configure hyperparameters and choose
which head(s) to train.

Example usage:
    python train_cub.py --data_dir data/processed_cub --batch_size 16 --max_epochs 10 \
                        --train_classification True --train_regression False
"""

import argparse
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from cub_datamodule import CUBDataModule


class TwoHeadedCUBModel(pl.LightningModule):
    """
    A small CNN with two heads:
      - classification head (outputs #classes logits)
      - regression head (outputs 4 bounding box coords)
    Both heads can be toggled on/off via arguments.
    """

    def __init__(
        self,
        num_classes=200,
        train_classification=True,
        train_regression=True,
        classification_weight=1.0,
        regression_weight=1.0,
        lr=1e-3,
    ):
        """
        :param num_classes: Number of bird classes in CUB (default=200).
        :param train_classification: Whether to train the classification head.
        :param train_regression: Whether to train the bbox regression head.
        :param classification_weight: Weight for the classification loss in total loss.
        :param regression_weight: Weight for the bounding box loss in total loss.
        :param lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.train_classification = train_classification
        self.train_regression = train_regression
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.lr = lr

        # Simple CNN backbone
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(
            (56, 56)
        )  # Global Average Pooling to reduce spatial dimensions

        # Classification head
        self.classifier = nn.Linear(32 * 56 * 56, num_classes)

        # Regression head
        self.bbox_regressor = nn.Linear(32 * 56 * 56, 4)

    def forward(self, x):
        # Shared backbone
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.gap(x)  # Global Average Pooling
        x = x.view(x.size(0), -1)

        x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization

        logits = self.classifier(x)
        bbox = self.bbox_regressor(x)
        return logits, bbox

    def training_step(self, batch, batch_idx):
        images, (labels, bboxes) = batch
        logits, bbox_preds = self(images)

        loss = 0.0
        log_dict = {}

        # Classification loss (cross entropy)
        if self.train_classification:
            classification_loss = F.cross_entropy(logits, labels)
            loss += self.classification_weight * classification_loss
            log_dict["train_classification_loss"] = classification_loss

        # Regression loss (MSE for bounding box)
        if self.train_regression:
            regression_loss = F.mse_loss(bbox_preds, bboxes)
            loss += self.regression_weight * regression_loss
            log_dict["train_regression_loss"] = regression_loss

        # Safety check
        if not self.train_classification and not self.train_regression:
            raise ValueError("Both classification and regression heads are disabled.")

        self.log_dict(log_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, (labels, bboxes) = batch
        logits, bbox_preds = self(images)

        loss = 0.0
        log_dict = {}

        if self.train_classification:
            classification_loss = F.cross_entropy(logits, labels)
            loss += classification_loss
            log_dict["val_classification_loss"] = classification_loss

        if self.train_regression:
            regression_loss = F.mse_loss(bbox_preds, bboxes)
            loss += regression_loss
            log_dict["val_regression_loss"] = regression_loss

        self.log_dict(log_dict, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)


def main():
    parser = argparse.ArgumentParser(
        description="Train a two-headed CNN on the CUB dataset."
    )
    # Data module args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_cub",
        help="Directory where train.csv, val.csv, images/ are located.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers."
    )

    # Model/training args
    parser.add_argument(
        "--num_classes",
        type=int,
        default=200,
        help="Number of bird classes in CUB (default=200).",
    )
    parser.add_argument(
        "--train_classification",
        type=bool,
        default=True,
        help="Whether to train the classification head.",
    )
    parser.add_argument(
        "--train_regression",
        type=bool,
        default=True,
        help="Whether to train the bounding box regression head.",
    )
    parser.add_argument(
        "--classification_weight",
        type=float,
        default=1.0,
        help="Weight for classification loss in the final sum.",
    )
    parser.add_argument(
        "--regression_weight",
        type=float,
        default=1.0,
        help="Weight for bounding box regression loss.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    # Trainer args
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Max training epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="EarlyStopping patience."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints_cub",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="Number of best checkpoints to keep (by monitored metric).",
    )

    args = parser.parse_args()

    # Initialize data module
    dm = CUBDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,  # or pass your custom transforms
    )
    dm.setup("fit")

    # Build model
    model = TwoHeadedCUBModel(
        num_classes=args.num_classes,
        train_classification=args.train_classification,
        train_regression=args.train_regression,
        classification_weight=args.classification_weight,
        regression_weight=args.regression_weight,
        lr=args.lr,
    )

    # Setup callbacks
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    # Decide which metric to monitor (if classification is on, watch classification loss)
    if args.train_classification:
        monitor_metric = "val_classification_loss"
    else:
        monitor_metric = "val_regression_loss"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=args.save_top_k,
        monitor=monitor_metric,
        filename="cub-{epoch:02d}-{" + monitor_metric + ":.4f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, patience=args.patience, mode="min"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
