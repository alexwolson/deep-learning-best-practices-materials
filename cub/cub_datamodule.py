# cub_datamodule.py

import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as T


class CUBDataset(Dataset):
    """
    A simple Dataset for the preprocessed CUB-200-2011 data.
    Expects a CSV with columns: filename, class_id, class_name, x_min, y_min, x_max, y_max.
    Loads images from <data_dir>/images/<filename>, and returns:
        - image (as a Tensor)
        - (class_id, bbox), where bbox = [x_min, y_min, x_max, y_max]
    """

    def __init__(self, csv_file, data_dir="data/processed_cub", transform=None):
        """
        :param csv_file: Path to the CSV file (train.csv or val.csv).
        :param data_dir: Base directory where the 'images/' folder resides.
        :param transform: Optional torchvision transform for the images.
        """
        super().__init__()
        self.csv_file = Path(csv_file)
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.samples = []
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # File path
        filename = row["filename"]
        img_path = self.data_dir / "images" / filename

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        # Apply transform (convert to Tensor, normalize, etc.)
        if self.transform:
            img = self.transform(img)

        # Parse class_id (convert to 0-based for PyTorch if desired)
        # e.g. if class_id runs 1..200, do int(row["class_id"]) - 1
        class_id = int(row["class_id"]) - 1  # 0-based

        # Parse bounding box
        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return img, (class_id, bbox)


class CUBDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for the preprocessed CUB dataset.
    Expects:
        - train.csv, val.csv in data_dir
        - images/ subfolder with resized images
    """

    def __init__(
        self,
        data_dir="data/processed_cub",
        batch_size=32,
        num_workers=4,
        transform=None,
    ):
        """
        :param data_dir: Directory with train.csv, val.csv, and images/.
        :param batch_size: Mini-batch size.
        :param num_workers: Number of multiprocessing workers for DataLoader.
        :param transform: Optional transform to apply to images (Tensor conversion, normalization, etc.).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # If no transform is provided, apply a default
        if transform is None:
            transform = T.ToTensor()
        self.transform = transform

    def setup(self, stage=None):
        """
        Called by Lightning to initialize datasets.
        """
        train_csv = self.data_dir / "train.csv"
        val_csv = self.data_dir / "val.csv"

        if stage == "fit" or stage is None:
            self.train_dataset = CUBDataset(
                csv_file=train_csv, data_dir=self.data_dir, transform=self.transform
            )
            self.val_dataset = CUBDataset(
                csv_file=val_csv, data_dir=self.data_dir, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
