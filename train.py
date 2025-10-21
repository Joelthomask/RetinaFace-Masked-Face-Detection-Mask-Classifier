import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.mobilenet_mask_classifier import get_model
from utils import save_checkpoint, plot_curves

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = 'data\Dataset'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "mobilenet_mask_best.pth.tar")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "mobilenet_mask_last.pth.tar")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Safe number of workers for Windows
NUM_WORKERS = 0 if platform.system() == "Windows" else 4

# -------------------------------
# TRANSFORMS
# -------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------
def main():
    # -------------------------------
    # DATASETS & LOADERS
    # -------------------------------
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # -------------------------------
    # MODEL, LOSS, OPTIMIZER
    # -------------------------------
    model = get_model(num_classes=2, pretrained=True, fine_tune=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------------
    # MIXED PRECISION SCALER
    # -------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=device=="cuda")

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        # ---- TRAIN ----
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device=="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.item())

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ---- VALIDATION ----
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=device=="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"GPU Memory Cached   : {torch.cuda.memory_reserved()/1024**2:.1f} MB")

        # ---- SAVE BEST MODEL ----
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc
            }, BEST_MODEL_PATH)
            print(f"✅ Saved new best model (Val Acc: {best_acc:.4f})")

        # ---- SAVE LAST MODEL ----
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc.item()
        }, LAST_MODEL_PATH)

    print(f"\n✅ Training complete! Best Val Acc: {best_acc:.4f}")

    # -------------------------------
    # PLOT CURVES
    # -------------------------------
    plot_curves(history, save_path=os.path.join(RESULTS_DIR, "training_curves.png"))

# -------------------------------
# ENTRY POINT (Windows Safe)
# -------------------------------
if __name__ == "__main__":
    main()
