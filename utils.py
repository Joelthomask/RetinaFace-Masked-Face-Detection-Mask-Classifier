import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save model checkpoint.
    """
    torch.save(state, filename)
    print(f"✅ Saved checkpoint: {filename}")


def load_checkpoint(filename, model, optimizer=None, device=None):
    """
    Load model checkpoint safely to the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"✅ Loaded checkpoint from {filename} on {device}")
    return model, optimizer


def plot_curves(history, save_path="results/training_curves.png"):
    """
    Plot training/validation loss and accuracy curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved training curves at {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png", labels=("Mask", "No Mask")):
    """
    Plot and save confusion matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Saved confusion matrix at {save_path}")
