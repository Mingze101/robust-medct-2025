# analyze_val_errors.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from models_robustmedct import build_resnet18, build_resnet34, IMG_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- Paths ----------
BASE_DIR = Path("IS_2025_OrganAMNIST")
TRAIN_LABELS_CSV = BASE_DIR / "train" / "labels_train.csv"
VAL_IMG_DIR = BASE_DIR / "val" / "images_val"
VAL_LABELS_CSV = BASE_DIR / "val" / "labels_val.csv"

FILE_COL = "file"
LABEL_COL = "label"
NUM_CLASSES = 11

# ---------- Transforms ----------
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# ---------- Dataset ----------
class RobustMedCTValDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        fname = row[FILE_COL]
        label = int(row[LABEL_COL])

        img_path = self.img_dir / fname
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


val_df = pd.read_csv(VAL_LABELS_CSV)
print("Val shape:", val_df.shape)

val_ds = RobustMedCTValDataset(val_df, VAL_IMG_DIR, transform=val_transform)
val_loader = DataLoader(
    val_ds,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ---------- Load checkpoints ----------
ckpt18 = torch.load("best_resnet18_robustmedct.pth", map_location=device)
ckpt34 = torch.load("best_resnet34_robustmedct.pth", map_location=device)
num_classes18 = ckpt18.get("num_classes", NUM_CLASSES)
num_classes34 = ckpt34.get("num_classes", NUM_CLASSES)
assert num_classes18 == num_classes34 == NUM_CLASSES

model18 = build_resnet18(num_classes=NUM_CLASSES, use_pretrained=False)
model18.load_state_dict(ckpt18["model_state"])
model18.to(device).eval()

model34 = build_resnet34(num_classes=NUM_CLASSES, use_pretrained=False)
model34.load_state_dict(ckpt34["model_state"])
model34.to(device).eval()

print("Models loaded.")

# ---------- Run inference on val ----------
all_labels = []
all_pred18 = []
all_pred34 = []
all_pred_ens = []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Val inference"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits18 = model18(imgs)
        logits34 = model34(imgs)

        probs18 = F.softmax(logits18, dim=1)
        probs34 = F.softmax(logits34, dim=1)
        probs_ens = (probs18 + probs34) / 2.0

        pred18 = probs18.argmax(dim=1)
        pred34 = probs34.argmax(dim=1)
        pred_ens = probs_ens.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_pred18.append(pred18.cpu().numpy())
        all_pred34.append(pred34.cpu().numpy())
        all_pred_ens.append(pred_ens.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_pred18 = np.concatenate(all_pred18)
all_pred34 = np.concatenate(all_pred34)
all_pred_ens = np.concatenate(all_pred_ens)

print("Total val samples:", len(all_labels))

# ---------- Helper: per-class stats ----------
def per_class_accuracy(y_true, y_pred, num_classes):
    acc = []
    counts = []
    for c in range(num_classes):
        mask = (y_true == c)
        n = mask.sum()
        counts.append(int(n))
        if n == 0:
            acc.append(0.0)
        else:
            acc.append((y_pred[mask] == c).mean())
    return np.array(acc), np.array(counts)

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

for name, preds in [("ResNet18", all_pred18),
                    ("ResNet34", all_pred34),
                    ("Ensemble18_34", all_pred_ens)]:
    acc, cnt = per_class_accuracy(all_labels, preds, NUM_CLASSES)
    overall = (all_labels == preds).mean()
    print(f"\n=== {name} ===")
    print(f"Overall acc: {overall:.4f}")
    print("Class | count | acc")
    for c in range(NUM_CLASSES):
        print(f"{c:2d}    {cnt[c]:5d}  {acc[c]:.4f}")

    cm = confusion_matrix_np(all_labels, preds, NUM_CLASSES)
    np.save(f"confmat_{name}.npy", cm)
    print(f"Confusion matrix saved to confmat_{name}.npy")
