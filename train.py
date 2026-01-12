# ============================================================
# BREAST CANCER MRI CLASSIFICATION
# EfficientNet | Stratified K-Fold | ROC per Fold | Clinical Metrics
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "train"
IMG_SIZE = 224
BATCH_SIZE = 30
EPOCHS = 20
FOLDS = 5
NUM_CLASSES = 2
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD DATASET
# =========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
labels = np.array(full_dataset.targets)

print("Class names:", full_dataset.classes)
print("Healthy images:", np.sum(labels == 0))
print("Sick images:", np.sum(labels == 1))

# =========================
# CLASS WEIGHTS
# =========================
class_counts = np.bincount(labels, minlength=NUM_CLASSES)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
)

print("Class weights:", class_weights)

# =========================
# STRATIFIED K-FOLD
# =========================
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

all_probs = []
all_targets = []
fold_aucs = []

plt.figure(figsize=(7, 6))

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n🔁 Fold {fold+1}/{FOLDS}")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_transform),
        val_idx
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # =========================
    # MODEL: EFFICIENTNET
    # =========================
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # =========================
    # TRAINING
    # =========================
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    fold_probs = []
    fold_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            fold_probs.extend(probs.cpu().numpy())
            fold_targets.extend(y.numpy())

    fpr, tpr, _ = roc_curve(fold_targets, fold_probs)
    fold_auc = auc(fpr, tpr)
    fold_aucs.append(fold_auc)

    plt.plot(fpr, tpr, label=f"Fold {fold+1} (AUC={fold_auc:.3f})")

    all_probs.extend(fold_probs)
    all_targets.extend(fold_targets)

# =========================
# ROC SUMMARY
# =========================
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Fold")
plt.legend()
plt.show()

print("\nMean AUC:", np.mean(fold_aucs))

# =========================
# FINAL METRICS
# =========================
preds = [1 if p >= 0.5 else 0 for p in all_probs]

print("\nClassification Report:")
print(classification_report(
    all_targets,
    preds,
    target_names=["Healthy", "Sick"]
))

cm = confusion_matrix(all_targets, preds)
TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("Confusion Matrix:")
print(cm)
print(f"Sensitivity (Recall – Sick): {sensitivity:.3f}")
print(f"Specificity (Recall – Healthy): {specificity:.3f}")

# =========================
# SAVE FINAL MODEL
# =========================
torch.save(model.state_dict(), "breast_cancer_efficientnet_final.pth")
print("\n✅ Model saved successfully")
