"""
data_inspection.py

Extended sanity + evaluation checks for Darcy surrogate datasets (64x64):

- Train/test distribution sanity (logK and h)
- Boundary-condition consistency checks on h
- Global MAE / RMSE
- Per-sample MAE (optioneel gesorteerd: worst first)
- Visual inspection of worst predictions
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG — MUST MATCH YOUR PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"

# Batch IDs (same split as train_unet.py)
TRAIN_BATCH_IDS = [0, 1, 2, 3, 4, 5]
TEST_BATCH_IDS = [8, 9]

# Normalization (same as train_unet.py)
LOGK_CENTER = 4.0
H_MEAN = 145.3243
H_STD = 35.5957

GRID_H = 64
GRID_W = 64

# Plot settings
SHOW_ORDERED_BY_ERROR = True     # worst samples first
MAX_ERROR_PLOTS = 5              # number of samples to visualize

# Load full train set or only test
LOAD_TRAIN_FULL = True

# Sampling for quantiles
MAX_SAMPLE_VALUES = 500_000


# ============================================================
# Helpers
# ============================================================

def load_batches(prefix: str, batch_ids: list[int]) -> np.ndarray:
    arrays = []
    for i in batch_ids:
        p = DATA_DIR / f"{prefix}_batch{i}.txt"
        if not p.exists():
            raise FileNotFoundError(p)
        a = np.loadtxt(p, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        arrays.append(a.reshape(-1, GRID_H, GRID_W))
    return np.concatenate(arrays, axis=0)


def sample_pixels(arr: np.ndarray, max_values: int, rng: np.random.Generator) -> np.ndarray:
    flat = arr.reshape(-1)
    if flat.size <= max_values:
        return flat.copy()
    idx = rng.choice(flat.size, size=max_values, replace=False)
    return flat[idx]


def summarize_field(name: str, arr: np.ndarray, sample_values: np.ndarray) -> None:
    flat = sample_values[np.isfinite(sample_values)]
    print(f"\n[{name}]")
    print(f"  shape: {arr.shape}")
    print(f"  mean/std: {flat.mean():.6f} / {flat.std():.6f}")
    print(f"  min/max:  {flat.min():.6f} / {flat.max():.6f}")
    q = np.quantile(flat, [0.01, 0.5, 0.99])
    print(f"  q01/q50/q99: {q[0]:.6f} / {q[1]:.6f} / {q[2]:.6f}")


def boundary_checks(h: np.ndarray, name: str) -> None:
    dirichlet = h[:, 0, :]
    top_grad = h[:, -1, :] - h[:, -2, :]
    right_grad = h[:, :, -1] - h[:, :, -2]
    left_grad = h[:, :, 1] - h[:, :, 0]

    print(f"\n[BC checks: {name}]")
    print(f"  Dirichlet row j=0 | mean={dirichlet.mean():.6f}, std-within={dirichlet.std(axis=1).mean():.6e}")
    print(f"  Top Neumann |dh/dy| mean={np.abs(top_grad).mean():.6e}")
    print(f"  Right Neumann |dh/dx| mean={np.abs(right_grad).mean():.6e}")
    print(f"  Left flux |dh/dx| mean={np.abs(left_grad).mean():.6e} (not ~0)")


def plot_triplet(logk, h_true, h_pred, title, levels=20):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title)

    im0 = axes[0].imshow(logk, origin="lower")
    axes[0].set_title("logK")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(h_true, origin="lower")
    cs1 = axes[1].contour(h_true, levels=levels, colors="k", linewidths=0.7)
    axes[1].set_title("true h")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(h_pred, origin="lower")
    cs2 = axes[2].contour(h_pred, levels=levels, colors="k", linewidths=0.7)
    axes[2].set_title("pred h")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2])

    err = h_pred - h_true
    im3 = axes[3].imshow(err, origin="lower")
    axes[3].set_title("pred − true")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(0)

    print("=== Data inspection + per-sample evaluation (64x64) ===")
    print("DATA_DIR:", DATA_DIR)

    # Load datasets
    k_test = load_batches("k_set_64x64", TEST_BATCH_IDS)
    h_test = load_batches("h_set_64x64", TEST_BATCH_IDS)

    if LOAD_TRAIN_FULL:
        k_train = load_batches("k_set_64x64", TRAIN_BATCH_IDS)
        h_train = load_batches("h_set_64x64", TRAIN_BATCH_IDS)
    else:
        k_train = h_train = None

    print("\n[Integrity]")
    print("  test :", k_test.shape, h_test.shape)
    if k_train is not None:
        print("  train:", k_train.shape, h_train.shape)

    # Distribution summaries
    print("\n[Distributions]")
    summarize_field("logK test", k_test, sample_pixels(np.log(k_test), MAX_SAMPLE_VALUES, rng))
    summarize_field("h test", h_test, sample_pixels(h_test, MAX_SAMPLE_VALUES, rng))

    if k_train is not None:
        summarize_field("logK train", k_train, sample_pixels(np.log(k_train), MAX_SAMPLE_VALUES, rng))
        summarize_field("h train", h_train, sample_pixels(h_train, MAX_SAMPLE_VALUES, rng))

    # Boundary conditions
    boundary_checks(h_test, "test")
    if h_train is not None:
        boundary_checks(h_train, "train")

    # ========================================================
    # Prediction evaluation
    # ========================================================

    pred_path = ROOT / "pred_test_unet_64x64.txt"
    if not pred_path.exists():
        print("\n[Prediction check] pred_test_unet_64x64.txt not found → skipped")
        return

    pred_norm = np.loadtxt(pred_path, dtype=np.float32).reshape(-1, GRID_H, GRID_W)
    y_test_norm = (h_test - H_MEAN) / H_STD

    err_norm = pred_norm - y_test_norm
    mse = np.mean(err_norm**2)
    mae = np.mean(np.abs(err_norm))

    print("\n[Prediction check]")
    print(f"  test RMSE (norm):  {math.sqrt(mse):.6f}")
    print(f"  test RMSE (units): {math.sqrt(mse) * H_STD:.6f}")
    print(f"  test MAE  (norm):  {mae:.6f}")
    print(f"  test MAE  (units): {mae * H_STD:.6f}")

    # Unnormalize predictions
    pred_test = pred_norm * H_STD + H_MEAN

    # Per-sample MAE
    sample_mae = np.mean(np.abs(pred_test - h_test), axis=(1, 2))

    if SHOW_ORDERED_BY_ERROR:
        order = np.argsort(sample_mae)[::-1]
    else:
        order = np.arange(len(sample_mae))

    print(f"\n[Plots] Showing up to {MAX_ERROR_PLOTS} samples")

    for rank, i in enumerate(order[:MAX_ERROR_PLOTS]):
        print(f"Sample {i:4d} | MAE = {sample_mae[i]:.4f} (rank {rank+1})")
        plot_triplet(
            np.log(k_test[i]),
            h_test[i],
            pred_test[i],
            title=f"Test sample {i} | MAE={sample_mae[i]:.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
