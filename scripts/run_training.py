import os
import yaml
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.env import is_colab
from src.models.efficientnet_b4 import create_model
from src.data.dataset import create_dataloaders_from_csv


# ---------------------------
# Load YAML configuration
# ---------------------------
def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Colab-specific setup
# ---------------------------
def setup_colab_paths(cfg):
    """
    If running in Colab:
    - mount Drive
    - symlink dataset folder
    - adjust output directory
    """
    from google.colab import drive
    drive.mount('/content/drive')

    # Create data/ symlink → real dataset on Drive
    data_path_drive = "/content/drive/MyDrive/ChestXRay"
    csv_path_drive = os.path.join(data_path_drive, "train.csv")
    img_dir_drive = data_path_drive

    # Symlink to project structure
    !rm -rf data
    !ln -s {img_dir_drive} data

    # Update config paths
    cfg["data"]["csv_path"] = "data/train.csv"
    cfg["data"]["data_dir"] = "data"

    # Output on Drive
    cfg["paths"]["output_dir"] = os.path.join(data_path_drive, "outputs")
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    return cfg


# ---------------------------
# Main Training Script
# ---------------------------
def main():

    cfg = load_config()

    # If running in Colab → adapt paths
    if is_colab():
        cfg = setup_colab_paths(cfg)

    # ------------------------
    # Create DataLoaders
    # ------------------------
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders_from_csv(
        csv_path=cfg["data"]["csv_path"],
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    # ------------------------
    # Create Model
    # ------------------------
    device = cfg["training"]["device"]
    model = create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"]
    )
    model = model.to(device)

    # ------------------------
    # Loss, Optimizer, Scheduler
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ------------------------
    # Training Loop (Fidelity Version)
    # ------------------------
    patience = cfg["training"]["early_stopping_patience"]
    best_val_acc = 0.0
    best_model = None
    epochs_no_improve = 0

    num_epochs = cfg["training"]["epochs"]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        train_loss, correct, total = 0.0, 0, 0

        # ------------------------
        # Train Phase
        # ------------------------
        for images, labels in tqdm(train_loader, desc=f"Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss /= total

        # ------------------------
        # Validation Phase
        # ------------------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        val_loss /= total

        print(f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # ------------------------
        # Scheduler
        # ------------------------
        scheduler.step(val_loss)

        # ------------------------
        # Early Stopping (by val_acc)
        # ------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

    # ------------------------
    # Save best model
    # ------------------------
    best_model_path = os.path.join(cfg["paths"]["output_dir"], "best_model.pth")
    torch.save(best_model, best_model_path)
    print(f"\n🎉 Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
