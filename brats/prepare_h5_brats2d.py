import re
import random
from pathlib import Path

import numpy as np
import h5py
import cv2

def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = x.mean()
    sigma = x.std() + 1e-8
    return (x - mu) / sigma

def save_npy(out_dir, split, idx, img, msk):
    img_dir = out_dir / split / "images"
    msk_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    np.save(img_dir / f"{idx:07d}.npy", img.astype(np.float32))
    np.save(msk_dir / f"{idx:07d}.npy", msk.astype(np.uint8))

def main(h5_dir, out_dir="brats2d", img_size=128, seed=42, val_ratio=0.1, test_ratio=0.1, min_mask_pixels=50,
         modality_index=3, mask_channel=0):
    random.seed(seed)
    np.random.seed(seed)

    h5_dir = Path(h5_dir)
    out_dir = Path(out_dir)

    files = sorted(h5_dir.glob("*.h5"))
    if not files:
        raise RuntimeError(f"No .h5 files found in: {h5_dir}")

    # Split by VOLUME id to avoid leakage
    vol_re = re.compile(r"volume_(\d+)_slice_(\d+)\.h5", re.IGNORECASE)
    vols = {}
    for f in files:
        m = vol_re.search(f.name)
        if not m:
            continue
        vid = int(m.group(1))
        vols.setdefault(vid, []).append(f)

    vol_ids = sorted(vols.keys())
    random.shuffle(vol_ids)

    n = len(vol_ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = set(vol_ids[:n_test])
    val_ids = set(vol_ids[n_test:n_test+n_val])
    train_ids = set(vol_ids[n_test+n_val:])

    print(f"Volumes: total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    counters = {"train":0, "val":0, "test":0}

    def split_of(vid):
        if vid in test_ids: return "test"
        if vid in val_ids: return "val"
        return "train"

    for vid, flist in vols.items():
        split = split_of(vid)
        for fpath in flist:
            with h5py.File(fpath, "r") as f:
                img = f["image"][()]   # (240,240,4)
                msk = f["mask"][()]    # (240,240,3)

            # pick modality + mask channel
            img2 = img[..., modality_index]
            mask2 = msk[..., mask_channel]

            if mask2.sum() < min_mask_pixels:
                continue

            img2 = zscore(img2)

            # resize
            img_r = cv2.resize(img2, (img_size, img_size), interpolation=cv2.INTER_LINEAR)[..., None]
            msk_r = cv2.resize(mask2.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)[..., None]

            save_npy(out_dir, split, counters[split], img_r, msk_r)
            counters[split] += 1

    print("Saved slices (kept):", counters)
    print("Output:", out_dir.resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--out_dir", default="brats2d")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--min_mask_pixels", type=int, default=50)
    ap.add_argument("--modality_index", type=int, default=3, help="0..3 (often 3=FLAIR)")
    ap.add_argument("--mask_channel", type=int, default=0, help="0..2 (often 0=WT)")
    args = ap.parse_args()
    main(**vars(args))