from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# Standard ImageNet normalization for pretrained vision backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    """
    Returns train and validation/test transformation pipelines.
    """
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for multi-class Brain MRI Classification.
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Consistent label mapping across all models
        self.classes = sorted(self.df["class_label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row["file_path"])
        
        # Load image via OpenCV and convert BGR -> RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[row["class_label"]]
        return image, torch.tensor(label_idx, dtype=torch.long), image_path


def create_dataloaders(df: pd.DataFrame, batch_size: int = 32, num_workers: int = 2, img_size: int = 224):
    """
    Creates stratified DataLoaders for Train, Validation, and Test partitions.
    """
    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_df = df[df["dataset_split"] == "train"]
    val_df = df[df["dataset_split"] == "val"]
    test_df = df[df["dataset_split"] == "test"]

    train_dataset = BrainMRIDataset(train_df, transform=train_transform)
    val_dataset = BrainMRIDataset(val_df, transform=eval_transform)
    test_dataset = BrainMRIDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes