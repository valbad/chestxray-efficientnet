import os
import argparse
import yaml
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from safetensors.torch import save_file

# === Imports from your project ===
from src.data.dataset import prepare_datasets
from src.model.efficientnet_b4 import create_model


# ============================================================
#   Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """Well-adapted for class-imbalanced classification."""
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
#   TRAINING FUNCTION
# ============================================================
def train(cfg, train_dir_override):
    """
    Main training pipeline.
    """
    cfg_data = cfg["data"]
    cfg_train = cfg["training"]
    cfg_model = cfg["model"]

    cfg_data["train_dir"] = train_dir_override

    print("📂 Using train_dir:", cfg_data["train_dir"])

    # ---------------------------------------------------------
    # Load datasets & loaders
    # ---------------------------------------------------------
    (
        train_df,
        val_df,
        train_dataset,
        val_dataset,
    ) = prepare_datasets(cfg_data)


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers = 8,
        persistent_workers=True
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers = 8,
        persistent_workers=True
    )
    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    device = torch.device(cfg_train["device"] if torch.cuda.is_available() else "cpu")
    print("🚀 Training on:", device)

    model = create_model(
        num_classes=cfg_model["num_classes"],
        pretrained=cfg_model["pretrained"],
    ).to(device)

    # ---------------------------------------------------------
    # Loss (Focal)
    # ---------------------------------------------------------
    class_counts = train_df["label"].value_counts(normalize=True).sort_index().values
    class_weights = 1.0 / class_counts
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=class_weights_tensor)

    # ---------------------------------------------------------
    # Optimizer / Scheduler
    # ---------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = GradScaler()

    # ---------------------------------------------------------
    # Training loop vars
    # ---------------------------------------------------------
    num_epochs = cfg_train["epochs"]
    patience = cfg_train["early_stopping_patience"]

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # ============================================================
    #                     MAIN TRAIN LOOP
    # ============================================================
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        # -------------------------
        #     TRAIN
        # -------------------------
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            running_total += labels.size(0)
            running_correct += outputs.argmax(1).eq(labels).sum().item()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # -------------------------
        #     VALIDATION
        # -------------------------
        model.eval()
        val_loss_tot, val_correct, val_total = 0.0, 0, 0

        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad(), autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss_tot += loss.item() * images.size(0)
            val_total += labels.size(0)
            val_correct += outputs.argmax(1).eq(labels).sum().item()

        val_loss = val_loss_tot / val_total
        val_acc = val_correct / val_total

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss {train_loss:.4f} | Acc {train_acc:.4f} || "
            f"Val Loss {val_loss:.4f} | Acc {val_acc:.4f}"
        )

        # Scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"⭐ New best val_acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

    # ============================================================
    #             SAVE BEST + LAST MODELS
    # ============================================================
    models_dir = os.path.join("outputs", "models")
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "efficientnet_b4_best.safetensors")
    save_file(best_model_state, best_model_path)
    print(f"✅ Best model saved at: {best_model_path}")

    last_model_path = os.path.join(models_dir, "efficientnet_b4_last_epoch.safetensors")
    save_file(model.state_dict(), last_model_path)
    print(f"📦 Last epoch model saved at: {last_model_path}")


# ============================================================
#   MAIN: CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 on chest X-ray data")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
    "--train_dir",
    type=str,
    required=True,
    help="Path to the training directory."
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(
        cfg,
        train_dir_override=args.train_dir
    )
