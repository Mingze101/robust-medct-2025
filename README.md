# RobustMedCT 2025: Robust Abdominal CT Classification

PyTorch implementation for 11-class organ classification from grayscale abdominal CT images. The project was developed for the Robust Medical Image Classification Challenge 2025 and focuses on robustness to image degradation, class imbalance, validation error analysis, and probability-level model ensembling.

## What is included

- ResNet-18 and ResNet-34 adapted to one-channel CT images
- Gaussian noise, salt-and-pepper noise, resolution degradation, affine augmentation, and random erasing
- Class-balanced sampling and validation-guided weighting of difficult classes
- Per-class validation accuracy and confusion-matrix analysis
- Two-model and four-model probability averaging
- Kaggle-compatible submission generation

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── models_robustmedct.py
├── train_resnet18.py
├── train_resnet34.py
├── analyze_val_errors.py
├── print_confusion_matrix.py
├── ensemble_2models.py
├── ensemble_4models.py
└── notebooks/
    └── robustmedct_experiments.ipynb
```

The dataset, trained checkpoints, NumPy outputs, and submission CSV files are intentionally not included.

## Dataset

Download the competition data from Kaggle:

https://www.kaggle.com/competitions/robust-med-ct-2025/data

Place it in the repository root using this structure:

```text
IS_2025_OrganAMNIST/
├── train/
│   ├── images_train/
│   └── labels_train.csv
├── val/
│   ├── images_val/
│   └── labels_val.csv
└── test/
    ├── images/
    └── manifest_public.csv
```

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

Train ResNet-18:

```bash
python train_resnet18.py
```

Train ResNet-34:

```bash
python train_resnet34.py
```

The training scripts save the best checkpoint according to validation macro-F1.

## Validation analysis

`analyze_val_errors.py` expects these local checkpoint names:

```text
best_resnet18_robustmedct.pth
best_resnet34_robustmedct.pth
```

If the selected checkpoints use version suffixes, create local copies with the expected names. For example, in PowerShell:

```powershell
Copy-Item best_resnet18_robustmedct_ver2.pth best_resnet18_robustmedct.pth
Copy-Item best_resnet34_robustmedct_ver2.pth best_resnet34_robustmedct.pth
```

Then run:

```bash
python analyze_val_errors.py
python print_confusion_matrix.py
```

The analysis creates confusion-matrix arrays locally. These generated `.npy` files are excluded from Git.

## Ensemble inference

Two-model ensemble:

```bash
python ensemble_2models.py
```

The script averages the ResNet-18 and ResNet-34 class probabilities.

Four-model ensemble:

```bash
python ensemble_4models.py
```

The four-model script expects these local checkpoint names:

```text
best_resnet18_robustmedct_ver2.pth
best_resnet18_robustmedct_ver3.pth
best_resnet34_robustmedct_ver2.pth
best_resnet34_robustmedct_ver3.pth
```

It writes `submission_ens4models.csv`, which is ignored by Git.

## Reproducibility notes

- Images are converted to grayscale and resized to 224 × 224.
- Models use a one-channel first convolution and an 11-class output layer.
- The training scripts initialize ResNet models from torchvision ImageNet weights.
- Check the competition rules before describing an ImageNet-initialized run as fully compliant with a no-external-data restriction.
- Validation-guided difficult-class weighting is a model-development choice and is not independent test evidence.

## Competition Result

| Leaderboard               |       Score |        Rank |
| ------------------------- | ----------: | ----------: |
| Final private leaderboard | **0.79589** | **19 / 27** |

Eight submissions were evaluated during the competition. The repository documents the training, validation-error analysis, and model-ensembling workflow developed for the challenge.

The result is reported directly from the final Kaggle private leaderboard. Model-specific scores are not reported because separate verified leaderboard records are unavailable.


## Disclaimer

This repository is for research and benchmarking only. It is not a clinical diagnostic system.

## License

Code is released under the MIT License. The competition dataset remains subject to the provider's and Kaggle's separate terms.
