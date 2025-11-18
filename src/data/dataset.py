# src/data/dataset.py

import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.data.transforms import build_transforms
from src.data.check_data import validate_dataset_structure
from src.data.download import ensure_data_dirs


class ChestXRayDataset(Dataset):
    """
    Train/Val dataset: uses a CSV (path,label) + image directory.
    """

    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.loc[idx, "path"]
        label = int(self.df.loc[idx, "label"])

        img_path = os.path.join(self.image_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class ChestXRayTestDataset(Dataset):
    """
    Test dataset: images only, no labels. Returns (image, filename).
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        exts = ("*.png", "*.jpg", "*.jpeg")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(image_dir, ext)))

        self.paths = sorted(files)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, filename


def prepare_datasets(cfg_data):
    """
    Pipeline complet de préparation des datasets:
      1) S'assure que data/train et data/test existent
      2) Télécharge depuis Dropbox si les dossiers sont vides
      3) Vérifie que train.csv et les images sont cohérents
      4) Split train/val
      5) Crée train, val, test datasets

    Args:
        cfg_data (dict): section "data" du default.yaml

    Returns:
        train_df, val_df, train_dataset, val_dataset, test_dataset
    """

    data_dir   = cfg_data["data_dir"]   # ex: "data"
    train_dir = cfg_data["train_dir"]
    test_dir   = cfg_data["test_dir"]   # ex: "data/test"
    csv_path   = cfg_data["train_csv"]  # ex: "data/train.csv"

    # 1–3 : gestion des dossiers + download si nécessaire
    ensure_data_dirs(cfg_data)

    # 4 : valider CSV + images de train
    df = validate_dataset_structure(csv_path, train_dir)

    # 4 : train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=cfg_data["val_size"],
        stratify=df["label"],
        random_state=cfg_data["random_state"],
    )

    # 5 : transformer et créer datasets
    transforms = build_transforms(cfg_data["img_size"])

    train_dataset = ChestXRayDataset(train_df, train_dir, transforms["train"])
    val_dataset   = ChestXRayDataset(val_df,   train_dir, transforms["val"])

    test_dataset = None
    # On ne force pas l'existence de data/test, mais si des images sont là, on les utilise
    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = ChestXRayTestDataset(test_dir, transforms["val"])

    return train_df, val_df, train_dataset, val_dataset, test_dataset
