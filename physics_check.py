"""
physics_check.py

Physics validation for U-Net predictions against Jeffrey's DGP (jeffrey_code.py).

What this checks (all aligned with jeffrey_code.py):
1) Basic accuracy vs ground-truth h (RMSE/MAE/relative L2).
2) Boundary conditions:
   - Bottom (j=0): Dirichlet h=100
   - Top (j=n-1): Neumann h_y = 0
   - Right (i=n-1): Neumann h_x = 0
   - Left (i=0): flux BC -k h_x = 500  (checked using harmonic face k)
3) Global mass balance:
      ∫_{∂Ω} q·n ds  ≈  ∫_{Ω} f dA
   with q = -k ∇h and harmonic face k on boundaries.
4) Discrete PDE residual:
      r = A(k) h_pred - b
   where A,b are constructed identically to jeffrey_code.solve_darcy_flow.
5) Projection-to-physics distance:
      Solve A δ = b - A h_pred, and report ||δ|| / ||h_true||.
6) Optional plots: true h / pred h / residual map.

Run from repo root:
  .\\.venv\\Scripts\\python.exe physics_check.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "datasets"

# Match your test split order (same as train_unet.py)
SET_IDS = ["_1000to1050", "_1050to1400"]

# Prediction file from train_unet.py export
model_id = "fc1"
PRED_FILE = f"models/{model_id}/pred_test.txt"

# Grid (same as jeffrey_code.py)
N = 60
L = 6.0
DX = L / (N - 1)

# Your updated normalization (you said you retrained with these)
PRED_IS_NORMALIZED = True
H_MEAN = 146
H_STD = 37

# How many samples for heavy checks (A build + sparse solve)
N_RESIDUAL_SAMPLES = 30
N_PROJECTION_SAMPLES = 10

# Plot a few qualitative examples
DO_PLOTS = True
N_PLOT_SAMPLES = 3

EPS = 1e-12


# -----------------------------
# Jeffrey source function (same logic as jeffrey_code.py)
# -----------------------------
def source_function(n: int) -> np.ndarray:
    dx = L / (n - 1)

    def source(x2: float) -> float:
        if 0.0 <= x2 <= 4.0:
            return 0.0
        elif 4.0 < x2 < 5.0:
            return 137.0
        else:  # 5 <= x2 <= 6
            return 274.0

    f = np.zeros(n * n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x2 = j * dx
            f[idx] = source(x2)
    return f


# -----------------------------
# IO helpers
# -----------------------------
def _load_txt(path: str) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    return a


def load_concat(prefix: str, set_ids: list[str]) -> np.ndarray:
    mats = []
    for sid in set_ids:
        path = os.path.join(DATASET_DIR, f"{prefix}{sid}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        mats.append(_load_txt(path))
    arr = np.concatenate(mats, axis=0)
    return arr.reshape((-1, N, N))


def load_predictions() -> np.ndarray:
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(f"Missing predictions file: {PRED_FILE}")

    pred = _load_txt(PRED_FILE).reshape((-1, N, N))
    if PRED_IS_NORMALIZED:
        pred = pred * H_STD + H_MEAN
    return pred


# -----------------------------
# Harmonic face k
# -----------------------------
def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (2.0 * a * b) / (a + b + EPS)


# -----------------------------
# Build A and b exactly like jeffrey_code.solve_darcy_flow
# -----------------------------
def build_system_matrix(kappa_flat: np.ndarray, f_flat: np.ndarray, n: int):
    dx = L / (n - 1)
    b = f_flat.astype(np.float64) * (dx ** 2)

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

            # interior
            if 0 < i < n - 1 and 0 < j < n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_N = k_face(i, j, i, j + 1)
                k_W = k_face(i, j, i - 1, j)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_N + k_E + k_S + k_W
                A[k, k + 1] = -k_E
                A[k, k - 1] = -k_W
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

            # left boundary: -k h_x(0,y) = 500
            if i == 0 and 0 < j < n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_N = k_face(i, j, i, j + 1)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_N + k_E + k_S
                A[k, k + 1] = -k_E
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

                b[k] += 500.0 * dx

            # right boundary: h_x(6,y) = 0
            if i == n - 1 and 0 < j < n - 1:
                k_W = k_face(i, j, i - 1, j)
                k_N = k_face(i, j, i, j + 1)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_N + k_W + k_S
                A[k, k - 1] = -k_W
                A[k, k + n] = -k_N
                A[k, k - n] = -k_S

            # lower boundary: h(x,0) = 100
            if j == 0 and 0 < i < n - 1:
                A[k, k] = 1.0
                b[k] = 100.0

            # upper boundary: h_y(x,6) = 0
            if j == n - 1 and 0 < i < n - 1:
                k_E = k_face(i, j, i + 1, j)
                k_W = k_face(i, j, i - 1, j)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_E + k_W + k_S
                A[k, k + 1] = -k_E
                A[k, k - 1] = -k_W
                A[k, k - n] = -k_S

            # corners
            if i == 0 and j == 0:  # lower left
                A[k, k] = 1.0
                b[k] = 100.0

            if i == n - 1 and j == 0:  # lower right
                A[k, k] = 1.0
                b[k] = 100.0

            if i == 0 and j == n - 1:  # upper left
                k_E = k_face(i, j, i + 1, j)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_E + k_S
                A[k, k + 1] = -k_E
                A[k, k - n] = -k_S

                b[k] += 500.0 * dx

            if i == n - 1 and j == n - 1:  # upper right
                k_W = k_face(i, j, i - 1, j)
                k_S = k_face(i, j, i, j - 1)

                A[k, k] = k_W + k_S
                A[k, k - 1] = -k_W
                A[k, k - n] = -k_S

    return A.tocsr(), b


# -----------------------------
# Physics checks
# -----------------------------
def boundary_checks(K: np.ndarray, h: np.ndarray) -> dict:
    """
    K, h: (samples, y, x) with y=j and x=i (same indexing as jeffrey_code.py)
    """

    # (1) Bottom Dirichlet: h(y=0) = 100 -> row j=0
    bot = h[:, 0, :]  # (samples, x)
    bot_abs_err = np.abs(bot - 100.0)
    bot_mean = bot_abs_err.mean()
    bot_max = bot_abs_err.max()

    # (2) Top Neumann: h_y(y=L) = 0 -> one-sided derivative at j=n-1
    hy_top = (h[:, -1, :] - h[:, -2, :]) / DX  # (samples, x)
    hy_top_mean = np.abs(hy_top[:, 1:-1]).mean()
    hy_top_max = np.abs(hy_top[:, 1:-1]).max()

    # (3) Right Neumann: h_x(x=L) = 0 -> one-sided derivative at i=n-1
    hx_right = (h[:, :, -1] - h[:, :, -2]) / DX  # (samples, y)
    hx_right_mean = np.abs(hx_right[:, 1:-1]).mean()
    hx_right_max = np.abs(hx_right[:, 1:-1]).max()

    # (4) Left flux: -k h_x(0,y) = 500 using harmonic face k between i=0 and i=1
    k_face_left = harmonic_mean(K[:, :, 0], K[:, :, 1])  # (samples, y)
    hx_left = (h[:, :, 1] - h[:, :, 0]) / DX             # (samples, y)
    qx_left = -k_face_left * hx_left                      # (samples, y), Darcy flux in +x direction

    qx_left_int = qx_left[:, 1:-1]  # exclude corners
    left_flux_mean_abs_err = np.abs(qx_left_int - 500.0).mean()
    left_flux_max_abs_err = np.abs(qx_left_int - 500.0).max()

    return {
        "bottom_dirichlet_mean_abs_err": float(bot_mean),
        "bottom_dirichlet_max_abs_err": float(bot_max),
        "top_neumann_mean_abs_hy": float(hy_top_mean),
        "top_neumann_max_abs_hy": float(hy_top_max),
        "right_neumann_mean_abs_hx": float(hx_right_mean),
        "right_neumann_max_abs_hx": float(hx_right_max),
        "left_flux_mean_abs_err_to_500": float(left_flux_mean_abs_err),
        "left_flux_max_abs_err_to_500": float(left_flux_max_abs_err),
    }


def global_mass_balance(K: np.ndarray, h: np.ndarray, f_flat: np.ndarray):
    """
    Checks net outward boundary flux ≈ total source integral.

    Uses q = -k ∇h with harmonic face k, and approximates boundary normal derivatives
    with one-sided differences between boundary nodes and adjacent interior nodes.

    Returns:
      diff: (samples,) array of (Q_out - S_total)
      rel : (samples,) array of diff / |S_total|
      S_total: scalar discrete integral sum(f)*dx^2
    """
    S_total = float(f_flat.sum() * (DX ** 2))

    # left boundary x=0, outward normal n=(-1,0) => q·n = -qx
    kL = harmonic_mean(K[:, :, 0], K[:, :, 1])        # (samples, y)
    dhdx_L = (h[:, :, 1] - h[:, :, 0]) / DX
    qx_L = -kL * dhdx_L
    qn_L = -qx_L

    # right boundary x=L, outward normal n=(+1,0) => q·n = +qx
    kR = harmonic_mean(K[:, :, -2], K[:, :, -1])      # (samples, y)
    dhdx_R = (h[:, :, -1] - h[:, :, -2]) / DX
    qx_R = -kR * dhdx_R
    qn_R = qx_R

    # bottom boundary y=0, outward normal n=(0,-1) => q·n = -qy
    kB = harmonic_mean(K[:, 0, :], K[:, 1, :])        # (samples, x)
    dhdy_B = (h[:, 1, :] - h[:, 0, :]) / DX
    qy_B = -kB * dhdy_B
    qn_B = -qy_B

    # top boundary y=L, outward normal n=(0,+1) => q·n = +qy
    kT = harmonic_mean(K[:, -2, :], K[:, -1, :])      # (samples, x)
    dhdy_T = (h[:, -1, :] - h[:, -2, :]) / DX
    qy_T = -kT * dhdy_T
    qn_T = qy_T

    # integrate along boundary: sum * dx
    Q_out = (qn_L.sum(axis=1) + qn_R.sum(axis=1) + qn_B.sum(axis=1) + qn_T.sum(axis=1)) * DX

    diff = Q_out - S_total
    rel = diff / (abs(S_total) + EPS)

    return diff, rel, S_total


def main():
    print("Loading datasets...")
    K = load_concat("k_set", SET_IDS)
    h_true = load_concat("h_set", SET_IDS)
    h_pred = load_predictions()

    n_samples = K.shape[0]
    if h_pred.shape[0] != n_samples:
        m = min(n_samples, h_pred.shape[0])
        print(f"[WARN] Prediction count ({h_pred.shape[0]}) != dataset count ({n_samples}). Using first {m}.")
        K = K[:m]
        h_true = h_true[:m]
        h_pred = h_pred[:m]
        n_samples = m

    print(f"Loaded {n_samples} samples. Grid: N={N}, L={L}, DX={DX:.6f}")

    # -------------------------
    # [0] Basic accuracy
    # -------------------------
    err = h_pred - h_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = np.linalg.norm(err.reshape(n_samples, -1), axis=1) / (np.linalg.norm(h_true.reshape(n_samples, -1), axis=1) + EPS)

    print("\n[0] Basic accuracy")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"Median relative L2 error per-sample: {np.median(rel_l2):.6f}")
    print(f"95th pct relative L2 error per-sample: {np.quantile(rel_l2, 0.95):.6f}")

    # Source vector
    f = source_function(N)

    # -------------------------
    # [1] Boundary checks
    # -------------------------
    print("\n[1] Boundary checks (True vs Pred)")
    bc_true = boundary_checks(K, h_true)
    bc_pred = boundary_checks(K, h_pred)
    for k in bc_true:
        print(f"{k:32s} | true: {bc_true[k]:.6f}  pred: {bc_pred[k]:.6f}")

    # -------------------------
    # [2] Global mass balance
    # -------------------------
    print("\n[2] Global mass balance: net outward boundary flux ≈ total source integral")
    diff_true, rel_true, S_total = global_mass_balance(K, h_true, f)
    diff_pred, rel_pred, _ = global_mass_balance(K, h_pred, f)

    print(f"Total source integral S = {S_total:.6f}")
    print(f"True: mean(Q_out - S) = {diff_true.mean():.6f}, mean|.| = {np.mean(np.abs(diff_true)):.6f}, mean rel = {rel_true.mean():.6e}")
    print(f"Pred: mean(Q_out - S) = {diff_pred.mean():.6f}, mean|.| = {np.mean(np.abs(diff_pred)):.6f}, mean rel = {rel_pred.mean():.6e}")

    # -------------------------
    # [3] Discrete PDE residual + projection distance (subset)
    # -------------------------
    print("\n[3] Discrete PDE residual r = A h - b (subset)")
    idxs = np.linspace(0, n_samples - 1, min(N_RESIDUAL_SAMPLES, n_samples), dtype=int)

    rel_res_true = []
    rel_res_pred = []
    proj_err = []

    for t, sidx in enumerate(idxs):
        kappa_flat = K[sidx].reshape(-1)
        A, b = build_system_matrix(kappa_flat, f, N)

        ht = h_true[sidx].reshape(-1)
        hp = h_pred[sidx].reshape(-1)

        rt = A @ ht - b
        rp = A @ hp - b

        denom = np.linalg.norm(b) + EPS
        rel_res_true.append(float(np.linalg.norm(rt) / denom))
        rel_res_pred.append(float(np.linalg.norm(rp) / denom))

        if t < min(N_PROJECTION_SAMPLES, len(idxs)):
            delta = spsolve(A, b - A @ hp)
            proj_err.append(float(np.linalg.norm(delta) / (np.linalg.norm(ht) + EPS)))

    print(f"True residual: mean rel ||r||/||b|| = {np.mean(rel_res_true):.6e}, median = {np.median(rel_res_true):.6e}")
    print(f"Pred residual: mean rel ||r||/||b|| = {np.mean(rel_res_pred):.6e}, median = {np.median(rel_res_pred):.6e}")

    if proj_err:
        print(f"Projection distance: mean ||delta||/||h_true|| = {np.mean(proj_err):.6e}, median = {np.median(proj_err):.6e}")

    # -------------------------
    # [4] Plots (optional)
    # -------------------------
    if DO_PLOTS:
        print("\n[4] Plotting a few samples (true / pred / residual)...")
        plot_idxs = np.linspace(0, n_samples - 1, min(N_PLOT_SAMPLES, n_samples), dtype=int)

        for sidx in plot_idxs:
            kappa_flat = K[sidx].reshape(-1)
            A, b = build_system_matrix(kappa_flat, f, N)

            rp = (A @ h_pred[sidx].reshape(-1) - b).reshape(N, N)

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].imshow(h_true[sidx], origin="lower")
            ax[0].set_title("True h")
            ax[1].imshow(h_pred[sidx], origin="lower")
            ax[1].set_title("Pred h")
            ax[2].imshow(rp, origin="lower")
            ax[2].set_title("Residual (A h_pred - b)")
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
