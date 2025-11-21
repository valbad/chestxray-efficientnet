import os
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import numpy as np
from PIL import Image
from torchvision import transforms

from src.data.dataset import ChestXRayDataset
from src.data.transforms import build_transforms
from src.models.efficientnet_b4 import create_model

def load_model(model_path, num_classes=3, device="cuda"):
    """Load a trained EfficientNet-B4 model."""
    model = create_model(num_classes=num_classes, pretrained=False)
    state = load_file(model_path)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

def gather_predictions(model, dataloader, device):
    """
    Predictions without TTA.
    Returns labels, preds, probabilities.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            preds  = probs.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            probs  = probs.cpu().numpy()

            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

def gather_predictions_tta(model, dataloader, device):
    """
    TTA for validation set.
    Accepts any batch size (batch_size >= 1).

    dataloader must return:
        img_stack: (B, K, C, H, W)
        labels:    (B,)
    """

    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for img_stack, labels in dataloader:

            # img_stack → (B, K, C, H, W)
            B, K, C, H, W = img_stack.shape

            # reshape to evaluate efficiently:
            # (B*K, C, H, W)
            img_reshaped = img_stack.view(B * K, C, H, W).to(device)

            logits = model(img_reshaped)                # (B*K, num_classes)
            probs = torch.softmax(logits, dim=1)        # (B*K, num_classes)

            # reshape back per-sample:
            # probs_per_sample: (B, K, num_classes)
            probs_per_sample = probs.view(B, K, -1)

            # mean over K augmentations
            probs_mean = probs_per_sample.mean(dim=1)   # (B, num_classes)

            preds = probs_mean.argmax(dim=1).cpu().numpy()

            # now collect
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs_mean.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def gather_predictions_no_labels(model, dataloader, device):
    """
    Predictions without TTA.
    Returns labels, preds, probabilities.
    """
    model.eval()
    all_paths, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            preds  = probs.argmax(dim=1).cpu().numpy()
            probs  = probs.cpu().numpy()

            all_paths.extend(list(paths))
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        all_paths,
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

def gather_predictions_tta_no_labels(model, dataloader, device):
    """
    TTA for classification.
    Each sample contains a stack of augmentations (K, C, H, W).
    """
    model.eval()
    all_paths, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for img_stack, paths in dataloader:

            # img_stack → (B, K, C, H, W)
            B, K, C, H, W = img_stack.shape

            # reshape to evaluate efficiently:
            # (B*K, C, H, W)
            img_reshaped = img_stack.view(B * K, C, H, W).to(device)

            logits = model(img_reshaped)                # (B*K, num_classes)
            probs = torch.softmax(logits, dim=1)        # (B*K, num_classes)

            # reshape back per-sample:
            # probs_per_sample: (B, K, num_classes)
            probs_per_sample = probs.view(B, K, -1)

            # mean over K augmentations
            probs_mean = probs_per_sample.mean(dim=1)   # (B, num_classes)

            preds = probs_mean.argmax(dim=1).cpu().numpy()

            # now collect
            all_paths.extend(list(paths))
            all_preds.append(preds)
            all_probs.append(probs_mean.cpu().numpy())
    
    return (
        all_paths, 
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )