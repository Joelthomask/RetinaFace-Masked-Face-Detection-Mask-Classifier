import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models.mobilenet_mask_classifier import get_model

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = 'data\Dataset'
BATCH_SIZE = 32
IMG_SIZE = 224
MODEL_PATH = "mobilenet_mask_best.pth"
CLASS_NAMES = ["Mask", "No_mask"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# TRANSFORMS
# -------------------------------
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------------
# DATASET & LOADER
# -------------------------------
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -------------------------------
# MODEL
# -------------------------------
model = get_model(num_classes=2, pretrained=False, fine_tune=False, device=device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# -------------------------------
# TEST LOOP
# -------------------------------
running_loss, running_corrects = 0.0, 0
all_preds, all_labels = [], []

print(f"\nRunning evaluation on {len(test_dataset)} images...\n")

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", leave=True):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------------------
# METRICS
# -------------------------------
test_loss = running_loss / len(test_dataset)
test_acc = running_corrects.double() / len(test_dataset)

print(f"\n=== RESULTS ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Precision, Recall, F1
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n=== CONFUSION MATRIX ===")
print(cm)
