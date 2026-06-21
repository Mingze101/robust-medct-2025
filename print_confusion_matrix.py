import numpy as np

cm = np.load("confmat_Ensemble18_34.npy")
num_classes = cm.shape[0]

for c in range(num_classes):
    row = cm[c]
    total = row.sum()
    print(f"\nTrue class {c}, total {total}")
    for p in range(num_classes):
        if row[p] > 0:
            print(f"  predicted {p}: {row[p]} ({row[p]/total:.3f})")
