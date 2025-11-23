import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def compute_metrics(all_labels, all_preds, all_probs, class_names):
    """ 
    Compute metrics: 
    - Accuracy
    - macro F1
    - Classification report 
    - Confusion matrix
    - Per-class ROC
    """
    metrics = {}

    # accuracy / f1
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")

    # classification report
    metrics["classification_report"] = classification_report(
        all_labels, all_preds, target_names=class_names
    )

    # confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds)

    # ROC curves
    num_classes = len(class_names)
    metrics["roc"] = []

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        metrics["roc"].append((fpr, tpr, auc(fpr, tpr)))

    # micro-average ROC
    y_bin = label_binarize(all_labels, classes=np.arange(num_classes))
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    metrics["roc_micro"] = (fpr_micro, tpr_micro, auc(fpr_micro, tpr_micro))

    return metrics

def plot_metrics(metrics, class_names, save_dir="assets/figures"):
    """ 
    Plot: 
    - Confusion matrix
    - Roc curves (per-class)
    - Save plots in save_dir folder
    """
    os.makedirs(save_dir, exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(
        metrics["confusion_matrix"], annot=True, fmt="d",
        cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.show()
    plt.close()

    # ROC curves
    plt.figure(figsize=(7,6))
    for i, cname in enumerate(class_names):
        fpr, tpr, auc_val = metrics["roc"][i]
        plt.plot(fpr, tpr, label=f"{cname} (AUC={auc_val:.3f})")

    fpr_micro, tpr_micro, auc_micro = metrics["roc_micro"]
    plt.plot(fpr_micro, tpr_micro, "k--", label=f"micro (AUC={auc_micro:.3f})")
    plt.plot([0,1], [0,1], "r--", label="random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png")
    plt.show()
    plt.close()
