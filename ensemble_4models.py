import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

from models_robustmedct import build_resnet18, build_resnet34, IMG_SIZE


# -------------------------
# Paths & basic config
# -------------------------
BASE_DIR = Path("IS_2025_OrganAMNIST")
TEST_IMG_DIR = BASE_DIR / "test" / "images"
TEST_MANIFEST_CSV = BASE_DIR / "test" / "manifest_public.csv"

BATCH_SIZE = 128
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Test transform (same as val_transform)
# -------------------------
test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# -------------------------
# Test dataset / dataloader
# -------------------------
class RobustMedCTTestDataset(Dataset):
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
        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, idx


test_df = pd.read_csv(TEST_MANIFEST_CSV)
print("Test manifest head:")
print(test_df.head())

test_ds = RobustMedCTTestDataset(
    test_df,
    TEST_IMG_DIR,
    transform=test_transform,
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

# -------------------------
# Load 4 checkpoints
# -------------------------
ckpt_specs = [
    ("best_resnet18_robustmedct_ver2.pth", build_resnet18, "ResNet18_v2"),
    ("best_resnet18_robustmedct_ver3.pth", build_resnet18, "ResNet18_v3"),
    ("best_resnet34_robustmedct_ver2.pth", build_resnet34, "ResNet34_v2"),
    ("best_resnet34_robustmedct_ver3.pth", build_resnet34, "ResNet34_v3"),
]

models = []
num_classes = None

for ckpt_path, builder, name in ckpt_specs:
    print(f"\nLoading checkpoint: {ckpt_path} ({name})")
    ckpt = torch.load(ckpt_path, map_location=device)

    n_cls = ckpt.get("num_classes", 11)
    if num_classes is None:
        num_classes = n_cls
    else:
        assert n_cls == num_classes, "num_classes mismatch between checkpoints"

    model = builder(num_classes=num_classes, use_pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    models.append((name, model))

print(f"\nLoaded {len(models)} models, num_classes = {num_classes}")

# -------------------------
# Helper: predict probabilities for one model
# -------------------------
def predict_proba(model, loader, desc="Predict proba"):
    all_probs = []
    with torch.no_grad():
        for imgs, indices in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)  # [N, num_classes]


# -------------------------
# Run inference for all 4 models
# -------------------------
probs_list = []

for name, model in models:
    print(f"\nRunning inference for {name} ...")
    probs = predict_proba(model, test_loader, desc=f"Predict {name}")
    probs_list.append(probs)

# -------------------------
# 4-model ensemble & build submission
# -------------------------
# Stack to shape [num_models, N, num_classes], then average over dim=0
probs_stack = torch.stack(probs_list, dim=0)
probs_ens = probs_stack.mean(dim=0)  # [N, num_classes]

preds_ens = probs_ens.argmax(dim=1).numpy()

indices = test_df["index"].to_numpy()
assert len(indices) == len(preds_ens)

submission = pd.DataFrame({
    "index": indices,
    "id": preds_ens.astype(int),
})

submission = submission.sort_values("index").reset_index(drop=True)
print(submission.head())
print(submission.tail())

out_path = "submission_ens4models.csv"
submission.to_csv(out_path, index=False)
print(f"4-model ensemble submission saved to: {out_path}")
