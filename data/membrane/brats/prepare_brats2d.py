import os
import glob
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
import random

def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mask = x != 0
    if mask.sum() < 10:
        return x
    mu = x[mask].mean()
    sigma = x[mask].std() + 1e-8
    x[mask] = (x[mask] - mu) / sigma
    return x

def save_split(out_dir, split, x_slices, y_slices):
    img_dir = out_dir / split / "images"
    msk_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, msk) in enumerate(zip(x_slices, y_slices)):
        np.save(img_dir / f"{i:06d}.npy", img.astype(np.float32))
        np.save(msk_dir / f"{i:06d}.npy", msk.astype(np.uint8))

def main(
    brats_train_dir,
    out_dir="brats2d",
    img_size=128,
    seed=42,
    val_ratio=0.1,
    test_ratio=0.1,
    min_edema_pixels=50,
):
    random.seed(seed)
    np.random.seed(seed)

    brats_train_dir = Path(brats_train_dir)
    out_dir = Path(out_dir)

    # Find cases by locating flair files
    flair_files = sorted(brats_train_dir.rglob("*_flair.nii.gz"))
    if len(flair_files) == 0:
        raise RuntimeError(f"No *_flair.nii.gz found under: {brats_train_dir}")

    cases = []
    for flair_path in flair_files:
        seg_path = Path(str(flair_path).replace("_flair.nii.gz", "_seg.nii.gz"))
        if seg_path.exists():
            cases.append((flair_path, seg_path))

    if len(cases) == 0:
        raise RuntimeError("Found flair files but no matching *_seg.nii.gz (are you pointing to Training set?)")

    random.shuffle(cases)

    n = len(cases)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_cases = cases[:n_test]
    val_cases = cases[n_test:n_test+n_val]
    train_cases = cases[n_test+n_val:]

    print(f"Cases: total={n}, train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    def process_cases(case_list):
        X, Y = [], []
        for flair_path, seg_path in case_list:
            flair = nib.load(flair_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()

            flair = zscore_nonzero(flair)

            # axial slices: [H, W, Z]
            z_slices = flair.shape[2]
            for z in range(z_slices):
                img = flair[:, :, z]
                mask = (seg[:, :, z] == 2).astype(np.uint8)  # edema label == 2

                if mask.sum() < min_edema_pixels:
                    continue

                # resize
                img_r = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                msk_r = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

                # add channel dim: (H,W,1)
                img_r = img_r[..., None]
                msk_r = msk_r[..., None]

                X.append(img_r)
                Y.append(msk_r)

        X = np.stack(X, axis=0) if len(X) else np.zeros((0, img_size, img_size, 1), dtype=np.float32)
        Y = np.stack(Y, axis=0) if len(Y) else np.zeros((0, img_size, img_size, 1), dtype=np.uint8)
        return X, Y

    Xtr, Ytr = process_cases(train_cases)
    Xva, Yva = process_cases(val_cases)
    Xte, Yte = process_cases(test_cases)

    print("Slices kept (edema-present):",
          f"train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")

    save_split(out_dir, "train", Xtr, Ytr)
    save_split(out_dir, "val", Xva, Yva)
    save_split(out_dir, "test", Xte, Yte)

    print("Done. Saved to:", out_dir.resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats_train_dir", required=True, help="Path to BraTS2020 Training directory (contains *_flair.nii.gz and *_seg.nii.gz)")
    ap.add_argument("--out_dir", default="brats2d", help="Output folder inside your repo (default: brats2d)")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--min_edema_pixels", type=int, default=50)
    args = ap.parse_args()
    main(**vars(args))