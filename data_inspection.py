"""
data_inspection.py

Quick sanity checks for Darcy surrogate datasets (64x64):
- Train/test distribution sanity (logK and h)
- Boundary-condition consistency checks on h
- Optional: evaluate saved predictions vs ground truth
- A few plots for human sanity
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
TEST_BATCH_IDS = [6]

# Normalization (same as train_unet.py)
LOGK_CENTER = 4.0
H_MEAN = 145.3243
H_STD = 35.5957

GRID_H = 64
GRID_W = 64

# Plot limits
N_PLOTS = 5

# If False → only test set is loaded
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
        if a.shape[1] != GRID_H * GRID_W:
            raise ValueError(f"{p} expected {GRID_H*GRID_W} cols, got {a.shape[1]}")
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
    ncols = 4 if h_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    fig.suptitle(title)

    # --- logK (NO contours) ---
    im0 = axes[0].imshow(logk, origin="lower")
    axes[0].set_title("logK")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0])

    # --- true h (WITH contours) ---
    im1 = axes[1].imshow(h_true, origin="lower")
    cs1 = axes[1].contour(
        h_true,
        levels=levels,
        colors="k",
        linewidths=0.7,
        origin="lower"
    )
    axes[1].clabel(cs1, inline=True, fontsize=7)
    axes[1].set_title("true h")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1])

    if h_pred is not None:
        # --- pred h (WITH contours) ---
        im2 = axes[2].imshow(h_pred, origin="lower")
        cs2 = axes[2].contour(
            h_pred,
            levels=levels,
            colors="k",
            linewidths=0.7,
            origin="lower"
        )
        axes[2].clabel(cs2, inline=True, fontsize=7)
        axes[2].set_title("pred h")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2])

        # --- error (NO contours) ---
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

    print("=== Data inspection (64x64) ===")
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
    s_logk_test = sample_pixels(np.log(k_test), MAX_SAMPLE_VALUES, rng)
    summarize_field("logK test", k_test, s_logk_test)

    s_h_test = sample_pixels(h_test, MAX_SAMPLE_VALUES, rng)
    summarize_field("h test", h_test, s_h_test)

    if k_train is not None:
        s_logk_train = sample_pixels(np.log(k_train), MAX_SAMPLE_VALUES, rng)
        summarize_field("logK train", k_train, s_logk_train)

        s_h_train = sample_pixels(h_train, MAX_SAMPLE_VALUES, rng)
        summarize_field("h train", h_train, s_h_train)

        print("\n[Train/Test shift]")
        print("  logK mean diff:", np.log(k_test).mean() - np.log(k_train).mean())
        print("  h mean diff   :", h_test.mean() - h_train.mean())

    # Normalization sanity
    print("\n[Normalization sanity]")
    print("  test y_norm mean:", (h_test.mean() - H_MEAN) / H_STD)
    print("  test y_norm std :", h_test.std() / H_STD)

    # BC checks
    boundary_checks(h_test, "test")
    if h_train is not None:
        boundary_checks(h_train, "train")

    # Optional predictions
    pred_test_path = ROOT / "pred_test_unet_64x64.txt"
    pred_test = None

    if pred_test_path.exists():
        pred_norm = np.loadtxt(pred_test_path, dtype=np.float32).reshape(-1, GRID_H, GRID_W)
        y_test_norm = (h_test - H_MEAN) / H_STD

        mse = np.mean((pred_norm - y_test_norm)**2)
        print("\n[Prediction check]")
        print("  test RMSE (norm):", math.sqrt(mse))
        print("  test RMSE (units):", math.sqrt(mse) * H_STD)

        pred_test = pred_norm * H_STD + H_MEAN
    else:
        print("\n[Prediction check] pred_test_unet_64x64.txt not found → skipped")

    # Plots
    print(f"\n[Plots] Showing up to {N_PLOTS} test samples")
    for i in range(min(N_PLOTS, k_test.shape[0])):
        plot_triplet(
            np.log(k_test[i]),
            h_test[i],
            pred_test[i] if pred_test is not None else None,
            title=f"Test sample {i}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
