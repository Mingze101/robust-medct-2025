#%% md
# # RobustMedCT_ECU
# 
# Sources: https://www.kaggle.com/competitions/robust-med-ct-2025/data
# 
# ### Dataset Description
# The dataset used in this challenge is a modified subset of the OrganAMNIST dataset, which originates from abdominal CT scans. Each image has been preprocessed into a small, standardized format to ensure fairness and accessibility for all participants.
# 
# #### Data Characteristics
# Modality: Grayscale abdominal CT images
# Resolution: 224 x224 pixels
# Classes: 11 distinct organ categories
# Labels: Each image is annotated with a class label (0–10), corresponding to a specific organ (Spleen, Right kidney, Left kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, Inferior vena cava, Pancreas, Right adrenal gland).
# File Format: Images provided as .png files (or .npy arrays, depending on release format) with labels in accompanying metadata files
# #### Data Splits
# Training Set: Includes images and their corresponding organ labels.
# Validation Set: Includes images and labels for model development and hyperparameter tuning.
# Test Set: Includes only images (no labels provided). This set may contain challenging or atypical cases, including images with subtle digital alterations or noise, simulating real-world deployment scenarios.
# #### Restrictions
# No external datasets are permitted.
# Participants must train and evaluate their models strictly on the data provided.
# #### Purpose
# This dataset has been intentionally designed to be compact yet challenging. While its resolution and size make it accessible to train models efficiently, the inclusion of atypical and digitally altered cases encourages participants to focus on robustness, generalization, and reliability rather than overfitting to clean data.
# 
# #### Link
# https://edithcowanuni-my.sharepoint.com/:f:/g/personal/m_alfawareh_ecu_edu_au/EuY7TFbOy8BDqLZHvCuHFPgBUu-qvArs4dBpgw2Ue7CcqA
#%% md
# #### Imports & paths & load CSVs
#%%
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Make plots show inline in Jupyter
#%matplotlib inline

# ------------------------------------------------------------------
# 1) Define base directories (assuming your notebook is in RobustMedCT_ECU/)
# ------------------------------------------------------------------
BASE_DIR = Path("IS_2025_OrganAMNIST")

TRAIN_IMG_DIR = BASE_DIR / "train" / "images_train"
TRAIN_LABELS_CSV = BASE_DIR / "train" / "labels_train.csv"

VAL_IMG_DIR = BASE_DIR / "val" / "images_val"
VAL_LABELS_CSV = BASE_DIR / "val" / "labels_val.csv"

TEST_IMG_DIR = BASE_DIR / "test" / "images"
TEST_MANIFEST_CSV = BASE_DIR / "test" / "manifest_public.csv"

print("Train images dir:", TRAIN_IMG_DIR)
print("Train labels CSV:", TRAIN_LABELS_CSV)
print("Val images dir:  ", VAL_IMG_DIR)
print("Val labels CSV:  ", VAL_LABELS_CSV)
print("Test images dir: ", TEST_IMG_DIR)
print("Test manifest:   ", TEST_MANIFEST_CSV)

# ------------------------------------------------------------------
# 2) Load CSV files
# ------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_LABELS_CSV)
val_df   = pd.read_csv(VAL_LABELS_CSV)
test_df  = pd.read_csv(TEST_MANIFEST_CSV)

print("\nTrain labels head:")
#display(train_df.head())
print(train_df.head())

print("\nVal labels head:")
#display(val_df.head())
print(val_df.head())
print("\nTest manifest head:")
#display(test_df.head())
print(test_df.head())
print("\nTrain shape:", train_df.shape,
      "| Val shape:", val_df.shape,
      "| Test shape:", test_df.shape)

#%%
# ------------------------------------------------------------------
# 3) Define which columns are file names and labels
# ------------------------------------------------------------------

print("Train CSV columns:", train_df.columns.tolist())

FILE_COL = "file"
LABEL_COL = "label"



if LABEL_COL not in train_df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in train_df.columns: {train_df.columns}")

# ------------------------------------------------------------------
# 4) Compute train label distribution (numeric class IDs only)
# ------------------------------------------------------------------
class_counts = train_df[LABEL_COL].value_counts().sort_index()

print("Train label distribution (counts):")
#display(class_counts)
print(class_counts)

plt.figure(figsize=(10, 4))
x = class_counts.index
y = class_counts.values

plt.bar(x, y)

# Use numeric class IDs as x-axis tick labels
xticklabels = [str(int(c)) for c in x]
plt.xticks(x, xticklabels, rotation=0)

plt.xlabel("Class ID")
plt.ylabel("Number of training samples")
plt.title("Training set label distribution (numeric classes)")
plt.tight_layout()
plt.show()

#%% md
#%% md
# #### Data Augmenting
#%%
import torchvision.transforms as T
import torch
import torch.nn.functional as F

IMG_SIZE = 224  # images are 224x224 according to the description

# -------------------------
# Custom corruption transforms
# -------------------------

class AddGaussianNoise(object):
    """
    Add Gaussian noise to a tensor image (values in [0, 1]).
    """
    def __init__(self, mean=0.0, std=0.05, p=0.7):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


class AddSaltPepperNoise(object):
    """
    Add salt-and-pepper noise to a tensor image.
    amount: percentage of pixels to corrupt (0.01 = 1%).
    s_vs_p: fraction of salt vs pepper.
    """
    def __init__(self, amount=0.01, s_vs_p=0.5, p=0.5):
        self.amount = amount
        self.s_vs_p = s_vs_p
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() >= self.p:
            return tensor

        c, h, w = tensor.shape
        num_pixels = int(self.amount * h * w)

        # salt
        num_salt = int(num_pixels * self.s_vs_p)
        if num_salt > 0:
            ys = torch.randint(0, h, (num_salt,), device=tensor.device)
            xs = torch.randint(0, w, (num_salt,), device=tensor.device)
            tensor[:, ys, xs] = 1.0

        # pepper
        num_pepper = num_pixels - num_salt
        if num_pepper > 0:
            ys = torch.randint(0, h, (num_pepper,), device=tensor.device)
            xs = torch.randint(0, w, (num_pepper,), device=tensor.device)
            tensor[:, ys, xs] = 0.0

        return tensor


class RandomDownsample(object):
    """
    Randomly downsample and upsample an image to simulate low-resolution /
    compression artifacts.
    scale_range: (min_scale, max_scale) relative to original size.
    """
    def __init__(self, scale_range=(0.4, 0.9), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() >= self.p:
            return tensor

        c, h, w = tensor.shape
        scale = torch.empty(1).uniform_(self.scale_range[0],
                                        self.scale_range[1]).item()
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # downsample then upsample back to original size
        x = tensor.unsqueeze(0)  # [1, C, H, W]
        x = F.interpolate(x, size=(new_h, new_w),
                          mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w),
                          mode="bilinear", align_corners=False)
        return x.squeeze(0)

# ----- hard classes & extra aug (tensor-based) -----
# Confusing classes from confusion matrix:
HARD_CLASSES = {1, 2, 4, 5, 7, 8}

# Extra augmentation applied only to hard classes, on tensor
extra_hard_transform = T.Compose([
    AddGaussianNoise(std=0.07, p=1.0),
    RandomDownsample(scale_range=(0.4, 0.8), p=1.0),
])
HARD_AUG_P = 0.5

# -------------------------
# Training-time transforms (heavy robustness)
# -------------------------
train_transform = T.Compose([
    # 1) Basic resizing
    T.Resize((IMG_SIZE, IMG_SIZE)),

    # 2) Geometric transforms (safe for left/right anatomy)
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=None
    ),

    # 3) To tensor in [0, 1]
    T.ToTensor(),

    # 4) Robustness-related corruptions on tensor
    T.RandomApply([AddGaussianNoise(std=0.05)], p=0.7),
    T.RandomApply([AddSaltPepperNoise(amount=0.01)], p=0.5),
    T.RandomApply([RandomDownsample(scale_range=(0.4, 0.9))], p=0.5),

    # 5) Stronger cutout / occlusion
    T.RandomErasing(
        p=0.5,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        value=0.0,
    ),

    # 6) Normalization
    T.Normalize(mean=[0.5], std=[0.5]),
])

# -------------------------
# Validation / test transforms (no heavy aug)
# -------------------------
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

print("Transforms with heavy robustness augmentation ready.")


#%% md
# #### Model Training
# 
#%%
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torch
import numpy as np

# Basic training params
BATCH_SIZE = 128
NUM_WORKERS = 4

# Make sure these match your CSV column names (should already be set in previous cells)
FILE_COL = "file"
LABEL_COL = "label"

# -------------------------
# Custom Dataset
# -------------------------
class RobustMedCTDataset(Dataset):
    """
    Simple dataset for the RobustMedCT_ECU challenge.
    Assumes:
      - df has columns [FILE_COL, LABEL_COL]
      - image files are located under img_dir / <file_name>
      - images are single-channel (grayscale)
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[FILE_COL]
        label = int(row[LABEL_COL])

        img_path = self.img_dir / fname
        img = Image.open(img_path).convert("L")  # grayscale

        if self.transform is not None:
            img = self.transform(img)

        # Extra hard-pair augmentation for confusing classes
        if label in HARD_CLASSES and torch.rand(1).item() < HARD_AUG_P:
            img = extra_hard_transform(img)

        return img, label


# -------------------------
# Class-balanced sampling with WeightedRandomSampler
# -------------------------
class_counts = train_df[LABEL_COL].value_counts().sort_index()
print("Class counts:\n", class_counts)

# Per-class validation accuracy of our current ensemble (from analyze_val_errors.py)
# We use these numbers to compute an "error boost" per class.
per_class_acc = {
    0: 0.9969,
    1: 0.8197,
    2: 0.7956,
    3: 0.9949,
    4: 0.8627,
    5: 0.8242,
    6: 1.0000,
    7: 0.9835,
    8: 0.9118,
    9: 0.9981,
    10: 0.9902,
}

gamma = 1.5  # how strongly we emphasize high-error classes (可以以后调)

base_class_weights = 1.0 / class_counts.values.astype(np.float32)

final_class_weights = []
for cls, base_w in zip(class_counts.index, base_class_weights):
    acc_c = per_class_acc.get(int(cls), 1.0)
    err_c = 1.0 - acc_c          # error rate
    boost = 1.0 + gamma * err_c  # >1 for hard classes, ~1 for easy ones
    final_class_weights.append(base_w * boost)

final_class_weights = np.asarray(final_class_weights, dtype=np.float32)

label_to_weight = {cls: w for cls, w in zip(class_counts.index, final_class_weights)}

sample_weights = train_df[LABEL_COL].map(label_to_weight).values.astype(np.float32)
sample_weights = torch.from_numpy(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,   # allow samples to be drawn multiple times per epoch
)


# -------------------------
# Create Datasets & DataLoaders
# -------------------------
train_ds = RobustMedCTDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_ds   = RobustMedCTDataset(val_df,   VAL_IMG_DIR,   transform=val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,   # use sampler for balanced sampling
    shuffle=False,     # must be False when sampler is used
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

num_classes = train_df[LABEL_COL].nunique()
print(f"Num classes: {num_classes}")
print("DataLoaders are ready.")

#%% md
# 
#%%
from torchvision import models
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
#from tqdm.notebook import tqdm
from tqdm.auto import tqdm

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Build model (ResNet34 -> 1-channel input + num_classes output)
# -------------------------
def build_resnet34(num_classes, use_pretrained=True):
    if use_pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet34(weights=None)
    
    # Change first conv to accept 1-channel input instead of 3
    w = model.conv1.weight.data
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    model.conv1.weight.data = w.mean(dim=1, keepdim=True)  # average over RGB
    
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def main():
    model = build_resnet34(num_classes=num_classes, use_pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # -------------------------
    # Helper: evaluate on validation set
    # -------------------------
    def evaluate(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, acc, f1

    # -------------------------
    # Training loop
    # -------------------------
    from torch.optim.lr_scheduler import CosineAnnealingLR
    NUM_EPOCHS = 60
    best_val_f1 = 0.0
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | "
              f"val_f1={val_f1:.4f}  | "
              f"time={epoch_time:.1f}s")

        # Save the best model based on validation macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"model_state": model.state_dict(), "num_classes": num_classes},
                "best_resnet34_robustmedct.pth",
            )
            print("  -> New best model saved.")
        # LR scheduler step
        scheduler.step()
    #%% md
    # #### Test Inference and Submission CSV FIle Generation
    #%%
    # ------ Test dataset & dataloader (updated) ------

    class RobustMedCTTestDataset(Dataset):
        """
        Test dataset (no labels).
        Expects a DataFrame with columns: 'index', 'file'.
        Returns (image_tensor, index_int).
        """
        def __init__(self, df, img_dir, transform=None,
                     index_col="index", file_col="file"):
            self.df = df.reset_index(drop=True)
            self.img_dir = Path(img_dir)
            self.transform = transform
            self.index_col = index_col
            self.file_col = file_col

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            idx = int(row[self.index_col])
            fname = row[self.file_col]

            img_path = self.img_dir / fname
            img = Image.open(img_path).convert("L")  # grayscale

            if self.transform is not None:
                img = self.transform(img)

            return img, idx

    print(test_df.head())

    test_ds = RobustMedCTTestDataset(
        test_df,
        TEST_IMG_DIR,
        transform=val_transform,  # use validation transform (no heavy aug)
        index_col="index",
        file_col="file",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("Num test images:", len(test_ds))

    #%%
    # ------ Load best checkpoint ------

    ckpt_path = "best_resnet34_robustmedct.pth"
    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes_ckpt = ckpt.get("num_classes", num_classes)
    print("num_classes in checkpoint:", num_classes_ckpt)


    model = build_resnet34(num_classes=num_classes_ckpt, use_pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    print("Checkpoint loaded.")

    # ------ Inference on test set ------

    all_indices = []
    all_preds = []

    with torch.no_grad():
        for imgs, indices in tqdm(test_loader, desc="Inference on test"):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()

            for idx_val, pred in zip(indices, preds):
                all_indices.append(int(idx_val))
                all_preds.append(int(pred))

    print("First 10 predictions:", list(zip(all_indices, all_preds))[:10])
    print("Total predictions:", len(all_preds))

    #%%
    # ------ Build submission DataFrame in the required format ------

    submission = pd.DataFrame({
        "index": all_indices,
        "id": all_preds,   # Kaggle wants column name 'id'
    })

    # Sort by index just to be safe
    submission = submission.sort_values("index").reset_index(drop=True)

    print(submission.head())
    print(submission.tail())

    submission_path = "submission.csv"  # required filename
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    train_df = pd.read_csv("IS_2025_OrganAMNIST/train/labels_train.csv")
    sub = pd.read_csv("submission.csv")

    print("Train label distribution (normalized):")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nPredicted test label distribution (normalized):")


    print(sub["id"].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    main()
