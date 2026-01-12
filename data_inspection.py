"""
data_inspection.py

Quick, high-ROI data checks for the Darcy-flow surrogate project:
- Train/test distribution sanity (logK and h)
- Boundary-condition consistency checks on h
- Optional: evaluate saved predictions (pred_test.txt / pred_train.txt) vs ground truth
- A few plots for human sanity (limited count; no plot spam)

Run:
  .\.venv\Scripts\python.exe data_inspection.py
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG (edit these if needed)
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"

# Use the SAME split as cnn.py
TRAIN_FILE_IDS = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]
TEST_FILE_IDS = ["_1000to1050", "_1050to1400"]

# Normalization used in cnn.py
LOGK_CENTER = 4.0
H_MEAN = 146.0
H_STD = 37.0

GRID_H = 60
GRID_W = 60

# Plot limits (avoid plot spam)
N_PLOTS = 6

# If your train set is huge and you only care about test shift checks, set False
LOAD_TRAIN_FULL = True

# Sampling for quantiles (approximate, but good enough)
MAX_SAMPLE_VALUES = 500_000  # number of pixel values to sample for quantile estimates


# -----------------------------
# Helpers
# -----------------------------
def _file_exists_or_warn(path: Path) -> bool:
    if path.exists():
        return True
    print(f"[WARN] Missing file: {path}")
    return False


def load_concat(prefix: str, file_ids: list[str]) -> np.ndarray:
    """
    Loads and concatenates multiple .txt arrays of shape (N, 3600) into (N_total, 60, 60) float32.
    """
    arrays = []
    for fid in file_ids:
        p = DATA_DIR / f"{prefix}{fid}.txt"
        if not _file_exists_or_warn(p):
            continue
        a = np.loadtxt(p).astype(np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != GRID_H * GRID_W:
            raise ValueError(f"{p} expected second dim {GRID_H*GRID_W}, got {a.shape[1]}")
        arrays.append(a.reshape(-1, GRID_H, GRID_W))
    if not arrays:
        raise FileNotFoundError(f"No files found for {prefix} with ids {file_ids}")
    return np.concatenate(arrays, axis=0)


def summarize_field(name: str, arr: np.ndarray, sample_values: np.ndarray | None = None) -> dict:
    """
    Print and return summary stats. If sample_values provided, use it for quantiles.
    """
    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    n = flat.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())

    mean = float(np.mean(flat[finite])) if n_finite > 0 else float("nan")
    std = float(np.std(flat[finite])) if n_finite > 0 else float("nan")
    vmin = float(np.min(flat[finite])) if n_finite > 0 else float("nan")
    vmax = float(np.max(flat[finite])) if n_finite > 0 else float("nan")

    if sample_values is None:
        # fallback: quantiles on full array (may be large)
        q = np.quantile(flat[finite], [0.01, 0.5, 0.99]) if n_finite > 0 else np.array([np.nan, np.nan, np.nan])
    else:
        finite_s = np.isfinite(sample_values)
        q = np.quantile(sample_values[finite_s], [0.01, 0.5, 0.99]) if finite_s.any() else np.array([np.nan, np.nan, np.nan])

    print(f"\n[{name}]")
    print(f"  shape: {arr.shape}")
    print(f"  count: {n} | finite: {n_finite} | nan: {n_nan} | inf: {n_inf}")
    print(f"  mean/std: {mean:.6f} / {std:.6f}")
    print(f"  min/max:  {vmin:.6f} / {vmax:.6f}")
    print(f"  q01/q50/q99 (approx): {q[0]:.6f} / {q[1]:.6f} / {q[2]:.6f}")

    return {"mean": mean, "std": std, "min": vmin, "max": vmax, "q01": float(q[0]), "q50": float(q[1]), "q99": float(q[2])}


def sample_pixels(arr: np.ndarray, max_values: int, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly sample up to max_values from arr (any shape).
    """
    flat = arr.reshape(-1)
    n = flat.size
    if n <= max_values:
        return flat.copy()
    idx = rng.choice(n, size=max_values, replace=False)
    return flat[idx].copy()


def boundary_checks(h: np.ndarray, name: str) -> None:
    """
    BC-consistency checks matching Jeffrey solver indexing:

    - Dirichlet fixed head is at j==0  => h[:, 0, :] should be constant (=100)
    - Upper boundary (j==n-1) no-flow  => dh/dy ~ 0 at top edge (row -1)
    - Right boundary (i==n-1) no-flow  => dh/dx ~ 0 at right edge (col -1)
    - Left boundary is flux BC         => dh/dx not necessarily ~ 0 (report magnitude only)
    """
    # h: (N, 60, 60)

    # Dirichlet edge (j=0)
    dirichlet = h[:, 0, :]  # (N, 60)
    dir_within_std = dirichlet.std(axis=1)
    dir_mean = dirichlet.mean(axis=1)

    # No-flow candidates (finite-difference normal gradients)
    top_grad = h[:, -1, :] - h[:, -2, :]          # approx dh/dy at j=n-1
    right_grad = h[:, :, -1] - h[:, :, -2]        # approx dh/dx at i=n-1
    left_grad = h[:, :, 1] - h[:, :, 0]           # flux BC (not ~0)

    top_abs = np.abs(top_grad).reshape(h.shape[0], -1).mean(axis=1)
    right_abs = np.abs(right_grad).reshape(h.shape[0], -1).mean(axis=1)
    left_abs = np.abs(left_grad).reshape(h.shape[0], -1).mean(axis=1)

    print(f"\n[BC checks: {name}]")
    print(f"  Dirichlet edge (j=0) mean across samples: mean={dir_mean.mean():.6f}, std={dir_mean.std():.6f}")
    print(f"  Dirichlet edge within-edge std (should be ~0): mean={dir_within_std.mean():.6f}, "
          f"q50={np.quantile(dir_within_std, 0.5):.6f}, q99={np.quantile(dir_within_std, 0.99):.6f}")

    print(f"  Top no-flow |dh/dn| approx (should be small): mean={top_abs.mean():.6f}, q99={np.quantile(top_abs, 0.99):.6f}")
    print(f"  Right no-flow |dh/dn| approx (should be small): mean={right_abs.mean():.6f}, q99={np.quantile(right_abs, 0.99):.6f}")

    print(f"  Left flux |dh/dx| approx (not ~0): mean={left_abs.mean():.6f}, q99={np.quantile(left_abs, 0.99):.6f}")


def plot_triplet(logk: np.ndarray, h_true: np.ndarray, h_pred: np.ndarray | None, title: str) -> None:
    """
    Plot one sample: logK, true h, pred h, and optionally error map.

    NOTE: origin='lower' so that array row 0 (j=0) appears at the bottom,
    matching the proposal statement "bottom boundary has h=100".
    """
    ncols = 4 if h_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    fig.suptitle(title)

    ax = axes[0]
    ax.set_title("log(K)")
    ax.imshow(logk, origin="lower")
    ax.axis("off")

    ax = axes[1]
    ax.set_title("true h")
    ax.imshow(h_true, interpolation="none", origin="lower")
    ax.contour(h_true, levels=20, colors=["black"])
    ax.axis("off")

    if h_pred is not None:
        ax = axes[2]
        ax.set_title("pred h")
        ax.imshow(h_pred, interpolation="none", origin="lower")
        ax.contour(h_pred, levels=20, colors=["black"])
        ax.axis("off")

        ax = axes[3]
        err = h_pred - h_true
        ax.set_title("pred - true")
        im = ax.imshow(err, interpolation="none", origin="lower")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    rng = np.random.default_rng(0)

    print("=== Data inspection ===")
    print("DATA_DIR:", DATA_DIR)

    # Load test always
    k_test = load_concat("k_set", TEST_FILE_IDS)   # (N_test, 60, 60)
    h_test = load_concat("h_set", TEST_FILE_IDS)   # (N_test, 60, 60)

    # Optionally load train
    if LOAD_TRAIN_FULL:
        k_train = load_concat("k_set", TRAIN_FILE_IDS)
        h_train = load_concat("h_set", TRAIN_FILE_IDS)
    else:
        k_train = None
        h_train = None

    # Basic integrity
    print("\n[Integrity]")
    print("  test:", k_test.shape, h_test.shape, "| k_nan:", np.isnan(k_test).any(), "| h_nan:", np.isnan(h_test).any())
    if k_train is not None:
        print("  train:", k_train.shape, h_train.shape, "| k_nan:", np.isnan(k_train).any(), "| h_nan:", np.isnan(h_train).any())

    # -----------------------------
    # Which edge is actually the fixed (constant) boundary?
    # -----------------------------
    h = h_test  # (N, 60, 60)

    edges = {
        "top":    h[:, 0, :],
        "bottom": h[:, -1, :],
        "left":   h[:, :, 0],
        "right":  h[:, :, -1],
    }

    print("\n[Edge diagnostics (to locate Dirichlet boundary)]")
    for name, edge in edges.items():
        edge_mean = float(edge.mean())
        # edge is always (N, 60) here, so std(axis=1) is within-edge variability per sample
        avg_within_std = float(edge.std(axis=1).mean())
        edge_min = float(edge.min())
        edge_max = float(edge.max())
        print(f"  {name:>6s}: mean={edge_mean:.3f}, avg-within-std={avg_within_std:.3f}, min={edge_min:.3f}, max={edge_max:.3f}")

    # Distribution summaries (logK and h)
    print("\n[Distribution summaries]")
    logk_test = np.log(k_test)
    s_logk_test = sample_pixels(logk_test, MAX_SAMPLE_VALUES, rng)
    summarize_field("logK test", logk_test, sample_values=s_logk_test)
    s_h_test = sample_pixels(h_test, MAX_SAMPLE_VALUES, rng)
    summarize_field("h test (raw units)", h_test, sample_values=s_h_test)

    if k_train is not None:
        logk_train = np.log(k_train)
        s_logk_train = sample_pixels(logk_train, MAX_SAMPLE_VALUES, rng)
        summarize_field("logK train", logk_train, sample_values=s_logk_train)
        s_h_train = sample_pixels(h_train, MAX_SAMPLE_VALUES, rng)
        summarize_field("h train (raw units)", h_train, sample_values=s_h_train)

        # Compare train/test shift quickly
        print("\n[Train/Test shift quick check]")
        print(f"  logK mean diff (test-train): {logk_test.mean() - logk_train.mean():.6f}")
        print(f"  logK std  diff (test-train): {logk_test.std()  - logk_train.std():.6f}")
        print(f"  h mean   diff (test-train):  {h_test.mean()   - h_train.mean():.6f}")
        print(f"  h std    diff (test-train):  {h_test.std()    - h_train.std():.6f}")

    # Normalization sanity (are constants reasonable for these splits?)
    print("\n[Normalization sanity]")
    print(f"  implied test y_norm mean: {(h_test.mean() - H_MEAN) / H_STD:.6f}")
    print(f"  implied test y_norm std:  {(h_test.std()) / H_STD:.6f}")
    if h_train is not None:
        print(f"  implied train y_norm mean: {(h_train.mean() - H_MEAN) / H_STD:.6f}")
        print(f"  implied train y_norm std:  {(h_train.std()) / H_STD:.6f}")

    # Boundary condition checks on h
    boundary_checks(h_test, "test")
    if h_train is not None:
        boundary_checks(h_train, "train")

    # Optional: evaluate predictions if present
    pred_test_path = ROOT / "pred_test.txt"
    pred_train_path = ROOT / "pred_train.txt"

    pred_test = None
    if pred_test_path.exists():
        pred_test_norm = np.loadtxt(pred_test_path).astype(np.float32)
        if pred_test_norm.ndim == 1:
            pred_test_norm = pred_test_norm[None, :]
        if pred_test_norm.shape[1] != GRID_H * GRID_W:
            raise ValueError(f"{pred_test_path} expected second dim {GRID_H*GRID_W}, got {pred_test_norm.shape[1]}")
        pred_test_norm = pred_test_norm.reshape(-1, GRID_H, GRID_W)

        # Compare to normalized truth
        y_test_norm = (h_test - H_MEAN) / H_STD

        # Metrics
        mse = float(np.mean((pred_test_norm - y_test_norm) ** 2))
        rmse_norm = math.sqrt(mse)
        rmse_units = rmse_norm * H_STD

        print("\n[Prediction check: pred_test.txt]")
        print(f"  test MSE (normalized):  {mse:.6f}")
        print(f"  test RMSE (normalized): {rmse_norm:.6f}")
        print(f"  test RMSE (head units): {rmse_units:.3f}")

        # Store unnormalized pred for plots
        pred_test = pred_test_norm * H_STD + H_MEAN
    else:
        print("\n[Prediction check]")
        print("  pred_test.txt not found (skip prediction evaluation).")

    # Plots: a few samples (no spam)
    print(f"\n[Plots] Showing up to {N_PLOTS} samples from TEST set...")
    n = min(k_test.shape[0], h_test.shape[0], N_PLOTS)
    for i in range(n):
        logk = np.log(k_test[i])
        h_true = h_test[i]
        h_pred = pred_test[i] if pred_test is not None and i < pred_test.shape[0] else None
        plot_triplet(logk, h_true, h_pred, title=f"Test sample i={i}")

    print("\nDone.")


if __name__ == "__main__":
    main()
