import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def crop_to_largest_foreground_region(pil_img, threshold=5):
    """
    Convert to grayscale, normalize to [0, 255], binarize, find largest contour
    and crop around it. If nothing is found, return original image.
    """
    img = np.array(pil_img.convert("L"))  # Convert to grayscale
    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255  # Normalize to [0, 255]
    img = img.astype(np.uint8)

    # Threshold: ignore very dark pixels (near 0)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return pil_img  # Return original if nothing found

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cropped = pil_img.crop((x, y, x + w, y + h))
    return cropped


def apply_clahe_to_pil(pil_img):
    """
    Apply CLAHE on a grayscale version of the PIL image,
    then normalize back to [0, 255].
    """
    img = np.array(pil_img.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img)

    cl_img = cv2.normalize(cl_img, None, 0, 255, cv2.NORM_MINMAX)

    return Image.fromarray(cl_img)


def custom_preprocessing_2(img, apply_clahe=True):
    """
    Original preprocessing: crop to largest foreground region,
    then optionally apply CLAHE.
    """
    img = crop_to_largest_foreground_region(img)
    if apply_clahe:
        img = apply_clahe_to_pil(img)
    return img


def build_transforms(img_size=380):
    """
    Rebuild the original transform dict:
      - custom_preprocessing_2
      - Resize
      - Data augmentation (train)
      - ToTensor
      - Normalize([0.5], [0.5]) for grayscale
    """
    train_transform = transforms.Compose([
        transforms.Lambda(custom_preprocessing_2),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(custom_preprocessing_2),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return {
        "train": train_transform,
        "val": val_transform,
    }

def build_tta_transforms(img_size):

    base = [
        transforms.Lambda(custom_preprocessing_2),          # crop + CLAHE
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]

    return [

        # 1) Vue originale (très important)
        transforms.Compose(base),

        # 2) Très léger jitter photométrique (safe)
        transforms.Compose([
            transforms.Lambda(custom_preprocessing_2),
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),

        # 3) Légère augmentation : sharpen / blur
        transforms.Compose([
            transforms.Lambda(custom_preprocessing_2),
            transforms.Resize((img_size, img_size)),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
    ]
