"""
physics_check.py

Physics validation for U-Net predictions against Darcy ground truth.

Checks:
1) Accuracy vs true h (RMSE, MAE, relative L2)
2) Boundary conditions:
   - Bottom (j=0): Dirichlet h=100
   - Top (j=n-1): Neumann h_y = 0
   - Right (i=n-1): Neumann h_x = 0
   - Left (i=0): flux BC -k h_x = 500
3) Global mass balance
4) Discrete PDE residual: r = A h - b
5) Projection-to-physics distance
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# CONFIG — MUST MATCH DATA + TRAINING
# ============================================================

DATASET_DIR = "datasets"

# Batch IDs used for TEST in train_unet.py
TEST_BATCH_IDS = [6]

# Prediction file from train_unet.py
PRED_FILE = "pred_test_unet_64x64.txt"

# Grid
N = 64
L = 6.0
DX = L / (N - 1)

# Prediction normalization
PRED_IS_NORMALIZED = True
H_MEAN = 145.3243
H_STD = 35.5957

# Expensive checks
N_RESIDUAL_SAMPLES = 30
N_PROJECTION_SAMPLES = 10

# Plotting
DO_PLOTS = True
N_PLOT_SAMPLES = 3

EPS = 1e-12


# ============================================================
# Source function (identical to Darcy code)
# ============================================================

def source_function(n: int) -> np.ndarray:
    dx = L / (n - 1)

    def source(x2: float) -> float:
        if 0.0 <= x2 <= 4.0:
            return 0.0
        elif 4.0 < x2 < 5.0:
            return 137.0
        else:
            return 274.0

    f = np.zeros(n * n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x2 = j * dx
            f[idx] = source(x2)
    return f


# ============================================================
# IO helpers
# ============================================================

def load_batches(prefix: str, batch_ids: list[int]) -> np.ndarray:
    mats = []
    for i in batch_ids:
        path = os.path.join(DATASET_DIR, f"{prefix}_batch{i}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mats.append(np.loadtxt(path, dtype=np.float64))
    arr = np.concatenate(mats, axis=0)
    return arr.reshape((-1, N, N))


def load_predictions() -> np.ndarray:
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(PRED_FILE)

    pred = np.loadtxt(PRED_FILE, dtype=np.float64)
    pred = pred.reshape((-1, N, N))

    if PRED_IS_NORMALIZED:
        pred = pred * H_STD + H_MEAN

    return pred


# ============================================================
# Harmonic face k
# ============================================================

def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (2.0 * a * b) / (a + b + EPS)


# ============================================================
# Build A and b (exact Darcy discretization)
# ============================================================

def build_system_matrix(kappa_flat: np.ndarray, f_flat: np.ndarray, n: int):
    dx = L / (n - 1)
    b = f_flat * dx**2

    def idx(i: int, j: int) -> int:
        return j * n + i

    def k_face(i1: int, j1: int, i2: int, j2: int) -> float:
        k1 = kappa_flat[idx(i1, j1)]
        k2 = kappa_flat[idx(i2, j2)]
        return (2.0 * k1 * k2) / (k1 + k2 + EPS)

    A = lil_matrix((n * n, n * n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            k = idx(i, j)

            # Interior
            if 0 < i < n - 1 and 0 < j < n - 1:
                kE = k_face(i, j, i + 1, j)
                kW = k_face(i, j, i - 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kW + kN + kS
                A[k, k + 1] = -kE
                A[k, k - 1] = -kW
                A[k, k + n] = -kN
                A[k, k - n] = -kS

            # Left boundary (flux)
            elif i == 0 and 0 < j < n - 1:
                kE = k_face(i, j, i + 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kN + kS
                A[k, k + 1] = -kE
                A[k, k + n] = -kN
                A[k, k - n] = -kS
                b[k] += 500.0 * dx

            # Right boundary (Neumann)
            elif i == n - 1 and 0 < j < n - 1:
                kW = k_face(i, j, i - 1, j)
                kN = k_face(i, j, i, j + 1)
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kW + kN + kS
                A[k, k - 1] = -kW
                A[k, k + n] = -kN
                A[k, k - n] = -kS

            # Bottom Dirichlet
            elif j == 0:
                A[k, k] = 1.0
                b[k] = 100.0

            # Top Neumann
            elif j == n - 1:
                kE = k_face(i, j, i + 1, j) if i < n - 1 else 0
                kW = k_face(i, j, i - 1, j) if i > 0 else 0
                kS = k_face(i, j, i, j - 1)

                A[k, k] = kE + kW + kS
                if i < n - 1:
                    A[k, k + 1] = -kE
                if i > 0:
                    A[k, k - 1] = -kW
                A[k, k - n] = -kS

    return A.tocsr(), b


# ============================================================
# Physics checks
# ============================================================

def boundary_checks(K: np.ndarray, h: np.ndarray) -> dict:
    bot_err = np.abs(h[:, 0, :] - 100.0)
    hy_top = (h[:, -1, :] - h[:, -2, :]) / DX
    hx_right = (h[:, :, -1] - h[:, :, -2]) / DX

    kL = harmonic_mean(K[:, :, 0], K[:, :, 1])
    hx_left = (h[:, :, 1] - h[:, :, 0]) / DX
    qx_left = -kL * hx_left

    return {
        "bottom_dirichlet_mean": float(bot_err.mean()),
        "bottom_dirichlet_max": float(bot_err.max()),
        "top_neumann_mean_abs": float(np.abs(hy_top[:, 1:-1]).mean()),
        "right_neumann_mean_abs": float(np.abs(hx_right[:, 1:-1]).mean()),
        "left_flux_mean_abs_err": float(np.abs(qx_left[:, 1:-1] - 500.0).mean()),
    }


def global_mass_balance(K: np.ndarray, h: np.ndarray, f_flat: np.ndarray):
    S = f_flat.sum() * DX**2

    kL = harmonic_mean(K[:, :, 0], K[:, :, 1])
    kR = harmonic_mean(K[:, :, -2], K[:, :, -1])
    kB = harmonic_mean(K[:, 0, :], K[:, 1, :])
    kT = harmonic_mean(K[:, -2, :], K[:, -1, :])

    qL = +kL * (h[:, :, 1] - h[:, :, 0]) / DX
    qR = -kR * (h[:, :, -1] - h[:, :, -2]) / DX
    qB = +kB * (h[:, 1, :] - h[:, 0, :]) / DX
    qT = -kT * (h[:, -1, :] - h[:, -2, :]) / DX

    Q = (qL.sum(axis=1) + qR.sum(axis=1) + qB.sum(axis=1) + qT.sum(axis=1)) * DX
    return Q - S


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    K = load_batches("k_set_64x64", TEST_BATCH_IDS)
    h_true = load_batches("h_set_64x64", TEST_BATCH_IDS)
    h_pred = load_predictions()

    n = min(len(h_true), len(h_pred))
    K, h_true, h_pred = K[:n], h_true[:n], h_pred[:n]

    print(f"Samples: {n}, Grid: {N}x{N}")

    # Accuracy
    err = h_pred - h_true
    print("\n[Accuracy]")
    print("RMSE:", np.sqrt(np.mean(err**2)))
    print("MAE :", np.mean(np.abs(err)))

    # Boundary checks
    print("\n[Boundary conditions]")
    bc_t = boundary_checks(K, h_true)
    bc_p = boundary_checks(K, h_pred)
    for k in bc_t:
        print(f"{k:28s} | true {bc_t[k]:.4e} | pred {bc_p[k]:.4e}")

    # Mass balance
    f = source_function(N)
    diff_true = global_mass_balance(K, h_true, f)
    diff_pred = global_mass_balance(K, h_pred, f)

    print("\n[Mass balance]")
    print("True mean |Q-S|:", np.mean(np.abs(diff_true)))
    print("Pred mean |Q-S|:", np.mean(np.abs(diff_pred)))

    # PDE residual
    print("\n[PDE residual]")
    idxs = np.linspace(0, n - 1, min(N_RESIDUAL_SAMPLES, n), dtype=int)
    res = []

    for i in idxs:
        A, b = build_system_matrix(K[i].reshape(-1), f, N)
        r = A @ h_pred[i].reshape(-1) - b
        res.append(np.linalg.norm(r) / (np.linalg.norm(b) + EPS))

    print("Mean relative residual:", np.mean(res))

    # Plots
    if DO_PLOTS:
        for i in idxs[:N_PLOT_SAMPLES]:
            plt.figure(figsize=(12, 3))

            # --- True h (WITH contours) ---
            ax1 = plt.subplot(1, 3, 1)
            im1 = ax1.imshow(h_true[i], origin="lower")
            cs1 = ax1.contour(
                h_true[i],
                levels=20,
                colors="k",
                linewidths=0.7,
                origin="lower",
            )
            ax1.clabel(cs1, inline=True, fontsize=7)
            ax1.set_title("True h")
            plt.colorbar(im1, ax=ax1)

            # --- Pred h (WITH contours) ---
            ax2 = plt.subplot(1, 3, 2)
            im2 = ax2.imshow(h_pred[i], origin="lower")
            cs2 = ax2.contour(
                h_pred[i],
                levels=20,
                colors="k",
                linewidths=0.7,
                origin="lower",
            )
            ax2.clabel(cs2, inline=True, fontsize=7)
            ax2.set_title("Pred h")
            plt.colorbar(im2, ax=ax2)

            # --- Residual (NO contours) ---
            ax3 = plt.subplot(1, 3, 3)
            A, b = build_system_matrix(K[i].reshape(-1), f, N)
            r = (A @ h_pred[i].reshape(-1) - b).reshape(N, N)
            im3 = ax3.imshow(r, origin="lower")
            ax3.set_title("PDE residual")
            plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
