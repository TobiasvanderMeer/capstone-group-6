"""
train_unet.py

Train a U-Net surrogate for Darcy flow:
- Input:  z = log(k) - 4  (N,1,60,60)
- Target: y = (h - 146)/37 (N,60,60)

Features:
- Physics-informed loss (Darcy law residual)
- Mixed precision (CPU/GPU safe)
- Gradient accumulation
- ReduceLROnPlateau scheduler
- Early stopping
- Optional Dirichlet enforcement (row j=0 fixed head)
- Batched prediction export
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler  # still works; GradScaler auto-disables on CPU

from unet import UNet60

# -----------------------------
# Config
# -----------------------------
TRAIN_FILE_IDS = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]
TEST_FILE_IDS = ["_1000to1050", "_1050to1400"]

LOGK_CENTER = 4.0
H_MEAN = 145.3243
H_STD = 35.5957

N_EPOCHS = 80
BATCH_SIZE = 16

BASE_CH = 64
LR = 3e-4
WEIGHT_DECAY = 1e-5
ENFORCE_DIRICHLET_ROW0 = True

PATIENCE = 12
MIN_DELTA = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4

PRED_BATCH_TRAIN = 16
PRED_BATCH_TEST = 64

SEED = 0

# -----------------------------
# Physics-informed loss
# -----------------------------
def physics_informed_loss(pred: torch.Tensor, y_true: torch.Tensor, k: torch.Tensor, lambda_phy: float = 0.1) -> torch.Tensor:
    """Physics-informed loss combining MSE and Darcy residual"""
    data_loss = nn.functional.mse_loss(pred, y_true)

    B, H, W = pred.shape
    dx = dy = 1.0

    pred_pad = nn.functional.pad(pred.unsqueeze(1), (1,1,1,1), mode='replicate')
    k_pad = nn.functional.pad(k, (1,1,1,1), mode='replicate')

    dh_dx = (pred_pad[:,:,1:-1,2:] - pred_pad[:,:,1:-1,:-2]) / (2*dx)
    dh_dy = (pred_pad[:,:,2:,1:-1] - pred_pad[:,:, :-2,1:-1]) / (2*dy)

    k_center = k_pad[:,:,1:-1,1:-1]
    k_dx = (k_pad[:,:,1:-1,2:] + k_center)/2
    k_dy = (k_pad[:,:,2:,1:-1] + k_center)/2

    flux_x = k_dx * dh_dx
    flux_y = k_dy * dh_dy

    flux_x_pad = nn.functional.pad(flux_x, (1,1,0,0), mode='replicate')
    flux_y_pad = nn.functional.pad(flux_y, (0,0,1,1), mode='replicate')
    div = (flux_x_pad[:,:,:,2:] - flux_x_pad[:,:,:,:-2]) / (2*dx) + \
          (flux_y_pad[:,:,2:,:] - flux_y_pad[:,:,:-2,:]) / (2*dy)

    physics_loss = torch.mean(div**2)
    return data_loss + lambda_phy * physics_loss

# -----------------------------
# Data helpers
# -----------------------------
def load_files(prefix_path: str, file_ids: list[str]) -> np.ndarray:
    parts = [np.loadtxt(prefix_path + fid + ".txt", dtype=np.float32) for fid in file_ids]
    return np.concatenate(parts, axis=0)

def predict_in_batches(model: nn.Module, z: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    n = z.shape[0]
    out = np.empty((n, 60, 60), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            zb = z[i:i + batch_size]
            pb = model(zb).squeeze(1).detach().float().cpu().numpy()
            out[i:i + pb.shape[0]] = pb
    return out

# -----------------------------
# Main training
# -----------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    x = torch.tensor(load_files("datasets/k_set", TRAIN_FILE_IDS).reshape((-1, 1, 60, 60)), dtype=torch.float32)
    y = torch.tensor(load_files("datasets/h_set", TRAIN_FILE_IDS).reshape((-1, 60, 60)), dtype=torch.float32)
    x_test = torch.tensor(load_files("datasets/k_set", TEST_FILE_IDS).reshape((-1, 1, 60, 60)), dtype=torch.float32)
    y_test = torch.tensor(load_files("datasets/h_set", TEST_FILE_IDS).reshape((-1, 60, 60)), dtype=torch.float32)

    # Normalize
    z = torch.log(x) - LOGK_CENTER
    y = (y - H_MEAN) / H_STD
    z_test = torch.log(x_test) - LOGK_CENTER
    y_test = (y_test - H_MEAN) / H_STD

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    z, y, z_test, y_test = z.to(device), y.to(device), z_test.to(device), y_test.to(device)

    # Model / optimizer / scheduler
    model = UNet60(base_ch=BASE_CH, enforce_dirichlet_row0=ENFORCE_DIRICHLET_ROW0).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    # Baseline
    mean_field = torch.mean(y, dim=0)
    baseline = torch.mean((y_test - mean_field) ** 2).item()
    print("baseline loss:", baseline)

    # Checkpoints
    out_dir = Path("checkpoints_unet")
    out_dir.mkdir(exist_ok=True)

    # Training setup
    scaler = torch.amp.GradScaler()  # auto disables on CPU
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_phy = 0.1
    grad_accum_steps = 2

    best_test, best_epoch, bad_epochs = float("inf"), 0, 0
    batch_idx = np.arange(z.shape[0])

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(batch_idx)

        epoch_losses = []
        n_batches = (z.shape[0] - 1) // BATCH_SIZE + 1
        optim.zero_grad(set_to_none=True)

        for b in range(n_batches):
            idx = batch_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            zb, yb = z[idx], y[idx]
            kb = torch.exp(zb + LOGK_CENTER)

            with torch.amp.autocast(device_type=device_type):
                pred = model(zb).squeeze(1)
                loss = physics_informed_loss(pred, yb, kb, lambda_phy)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (b + 1) % grad_accum_steps == 0 or (b + 1) == n_batches:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            epoch_losses.append(loss.item() * grad_accum_steps)

        train_loss = float(np.mean(epoch_losses))

        # Test evaluation
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=device_type):
            kb_test = torch.exp(z_test + LOGK_CENTER)
            pred_test = model(z_test).squeeze(1)
            test_loss = physics_informed_loss(pred_test, y_test, kb_test, lambda_phy).item()

        scheduler.step(test_loss)
        dt = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]
        print(f"epoch {epoch:>3}/{N_EPOCHS} | lr {lr_now:.2e} | train {train_loss:.6f} | test {test_loss:.6f} | {dt:.1f}s")

        # Early stopping
        if test_loss < best_test - MIN_DELTA:
            best_test, best_epoch, bad_epochs = test_loss, epoch, 0
            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optim_state_dict": optim.state_dict()}
            torch.save(ckpt, out_dir / "unet_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, best test {best_test:.6f})")
                break

    # -------------------------
    # Export predictions
    # -------------------------
    best = torch.load(out_dir / "unet_best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred_train = predict_in_batches(model, z, batch_size=PRED_BATCH_TRAIN)
    pred_test = predict_in_batches(model, z_test, batch_size=PRED_BATCH_TEST)

    np.savetxt("pred_train_unet.txt", pred_train.reshape((-1, 3600)))
    np.savetxt("pred_test_unet.txt", pred_test.reshape((-1, 3600)))

    print("saved:", "pred_train_unet.txt, pred_test_unet.txt,", str(out_dir / "unet_best.pt"))
    print(f"best test loss: {best_test:.6f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
