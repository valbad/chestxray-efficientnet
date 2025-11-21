# src/data/dataset.py

import os
import glob
import pandas as pd
from PIL import Image
import torch
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

class TTADataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms_list):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.loc[idx, "path"])
        img = Image.open(img_path).convert("RGB")
        augmented = [t(img.convert("L")) for t in self.transforms_list]
        label = int(self.df.loc[idx, "label"])
        return torch.stack(augmented), label

class TTATestDataset(Dataset):
    """
    TTA dataset for test images (no labels).
    Returns:
       (stack_of_augmentations, filename)
    """

    def __init__(self, image_dir, transforms_list):
        self.image_dir = image_dir
        self.transforms_list = transforms_list

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

        img = Image.open(img_path).convert("RGB")

        # apply the TTA transforms
        augmented = [t(img.convert("L")) for t in self.transforms_list]

        return torch.stack(augmented), filename

def prepare_datasets(cfg_data):
    """
    Pipeline complet de préparation des datasets:
      1) S'assure que data/train et data/test existent
      2) Télécharge depuis Dropbox si les dossiers sont vides
      3) Vérifie que train.csv et les images sont cohérents
      4) Split train/val
      5) Crée train, val, test datasets + test_df

    Returns:
        train_df, val_df, test_df,
        train_dataset, val_dataset, test_dataset
    """

    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_dir   = cfg_data["data_dir"]
    train_dir  = cfg_data["train_dir"]
    test_dir   = cfg_data["test_dir"]
    csv_path   = cfg_data["train_csv"]

    # -------------------------------
    # 1–3. Ensure data is present + download if needed
    # -------------------------------
    ensure_data_dirs(cfg_data)

    # -------------------------------
    # 4. Validate train CSV + images
    # -------------------------------
    df = validate_dataset_structure(csv_path, train_dir)

    # -------------------------------
    # 5. Train/Val split (with reset_index)
    # -------------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=cfg_data["val_size"],
        stratify=df["label"],
        random_state=cfg_data["random_state"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # -------------------------------
    # 6. Build transforms
    # -------------------------------
    transforms = build_transforms(cfg_data["img_size"])

    # -------------------------------
    # 7. Create train & val datasets
    # -------------------------------
    train_dataset = ChestXRayDataset(train_df, train_dir, transforms["train"])
    val_dataset   = ChestXRayDataset(val_df,   train_dir, transforms["val"])

    # -------------------------------
    # 8. Build test_df and test_dataset
    # -------------------------------
    test_df = None
    test_dataset = None

    if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        # créer dataframe test_df
        test_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        test_df = pd.DataFrame({"path": test_files})
        test_df = test_df.reset_index(drop=True)

        test_dataset = ChestXRayTestDataset(test_dir, transforms["val"])

    return train_df, val_df, test_df, train_dataset, val_dataset, test_dataset
