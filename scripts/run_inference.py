#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.inference.predict import (
    load_model,
    gather_predictions_no_labels,
    gather_predictions_tta_no_labels,
)
from src.data.dataset import ChestXRayTestDataset, TTATestDataset
from src.data.transforms import build_transforms, build_tta_transforms
from src.data.dataset import prepare_dataset_test


# ---------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on chest X-ray model.")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (same as used for training).")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the .safetensors model file.")

    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test images.")

    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "tta"],
                        help="Inference mode: 'normal' or 'tta'.")

    parser.add_argument("--save_dir", type=str, default="outputs/predictions",
                        help="Directory to store CSV results.")

    return parser.parse_args()


# ---------------------------------------------------------
# MAIN INFERENCE FUNCTION
# ---------------------------------------------------------
def run_inference():
    args = parse_args()

    # ----- Load config -----
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_data = cfg["data"]
    img_size = cfg_data["img_size"]

    # ----- Device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running inference on:", device)

    # ----- Load model -----
    model = load_model(
        args.model_path,
        num_classes=cfg["model"]["num_classes"],
        device=device
    )

    # -----------------------------------------------------
    # Build test_df + test dataset (standard)
    # -----------------------------------------------------
    cfg_data["test_dir"] = args.test_dir
    test_df, test_dataset = prepare_dataset_test(cfg_data)
    print(f"Found {len(test_df)} test images.")

    # -----------------------------------------------------
    # Mode = NORMAL
    # -----------------------------------------------------
    if args.mode == "normal":
        print("🔎 Running NORMAL inference...")

        tfms = build_transforms(img_size)["val"]

        loader = DataLoader(
            test_dataset,
            batch_size=cfg_data["batch_size"],
            shuffle=False,
            num_workers=cfg_data["num_workers"],
            pin_memory=True
        )

        all_paths, all_preds, all_probs = gather_predictions_no_labels(
            model,
            loader,
            device
        )

        # Save CSV
        os.makedirs(args.save_dir, exist_ok=True)
        out_csv = os.path.join(args.save_dir, "predictions.csv")

        df = pd.DataFrame({
            "path": all_paths,
            "label": all_preds,
            "prob0": all_probs[:, 0],
            "prob1": all_probs[:, 1],
            "prob2": all_probs[:, 2],
        })
        df.to_csv(out_csv, index=False)

        print(f"✅ Predictions saved to {out_csv}")
        return

    # -----------------------------------------------------
    # Mode = TTA
    # -----------------------------------------------------
    if args.mode == "tta":
        print("🔎 Running TTA inference...")

        tta_tfms = build_tta_transforms(img_size)

        test_dataset = TTATestDataset(
            image_dir=args.test_dir,
            transforms_list=tta_tfms
        )

        loader = DataLoader(
            test_dataset,
            batch_size=cfg_data["batch_size"],
            shuffle=False,
            num_workers=cfg_data["num_workers"],
            pin_memory=True
        )

        all_paths, all_preds, all_probs = gather_predictions_tta_no_labels(
            model,
            loader,
            device
        )

        # Save CSV
        os.makedirs(args.save_dir, exist_ok=True)
        out_csv = os.path.join(args.save_dir, "predictions_tta.csv")

        df = pd.DataFrame({
            "path": all_paths,
            "label": all_preds,
            "prob0": all_probs[:, 0],
            "prob1": all_probs[:, 1],
            "prob2": all_probs[:, 2],
        })
        df.to_csv(out_csv, index=False)

        print(f"✅ TTA predictions saved to {out_csv}")
        return


# ---------------------------------------------------------
if __name__ == "__main__":
    run_inference()
