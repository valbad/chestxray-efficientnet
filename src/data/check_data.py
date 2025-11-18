# src/data/check_data.py

import os
import pandas as pd


class DataValidationError(Exception):
    """Custom exception raised when dataset structure is invalid."""
    pass


def validate_dataset_structure(csv_path, image_dir):
    """
    Validates that:
      - train.csv exists
      - CSV has 2 columns: path,label
      - image dir exists
      - every image referenced in CSV exists in image_dir

    Returns:
      df: dataframe with columns ["path", "label"]
    """

    # ---------------------------------------------
    # 1. Check CSV
    # ---------------------------------------------
    if not os.path.isfile(csv_path):
        raise DataValidationError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        raise DataValidationError(f"Could not read CSV file:\n{e}")

    if df.shape[1] != 2:
        raise DataValidationError("CSV must contain exactly 2 columns: path,label")

    df.columns = ["path", "label"]

    # ---------------------------------------------
    # 2. Check images directory
    # ---------------------------------------------
    if not os.path.isdir(image_dir):
        raise DataValidationError(f"Image directory not found: {image_dir}")

    # ---------------------------------------------
    # 3. Check all referenced images exist
    # ---------------------------------------------
    missing = []
    for rel_path in df["path"]:
        img_path = os.path.join(image_dir, rel_path)
        if not os.path.isfile(img_path):
            missing.append(img_path)

    if missing:
        raise DataValidationError(
            f"Missing images referenced in CSV ({len(missing)} missing). "
            f"First missing examples:\n" +
            "\n".join(missing[:10]) +
            ("\n...(truncated)" if len(missing) > 10 else "")
        )

    # ---------------------------------------------
    # 4. Check labels are integers
    # ---------------------------------------------
    if not pd.api.types.is_integer_dtype(df["label"]):
        raise DataValidationError("Labels in CSV must be integers.")

    return df
