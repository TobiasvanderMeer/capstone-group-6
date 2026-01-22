"""
train_unet.py

Train a U-Net surrogate for Darcy flow (64x64):
- Input:  z = log(k) - 4      (N,1,64,64)
- Target: y = (h - mean)/std (N,64,64)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn



# Config


# Batch files: datasets/k_set_64x64_batch{i}.txt
TRAIN_BATCH_IDS = list(range(0, 6))  
TEST_BATCH_IDS  = list(range(6, 8))  


LOGK_CENTER = 4.0
H_MEAN = 145.3243
H_STD = 35.5957

N_EPOCHS = 80
BATCH_SIZE = 16

BASE_CH = 64
LR = 3e-4
WEIGHT_DECAY = 1e-5
ENFORCE_DIRICHLET_ROW0 = True

# Early stopping
PATIENCE = 12
MIN_DELTA = 1e-4

# Scheduler
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 4

# Prediction batching
PRED_BATCH_TRAIN = 16
PRED_BATCH_TEST = 32

SEED = 0



# Data loading


def load_batches(prefix: str, batch_ids: list[int]) -> np.ndarray:
    parts = []
    for i in batch_ids:
        fname = f"datasets/{prefix}_batch{i}.txt"
        print("loading", fname)
        parts.append(np.loadtxt(fname, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def predict_in_batches(model: nn.Module, z: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    n = z.shape[0]
    out = np.empty((n, 64, 64), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            zb = z[i:i + batch_size]
            pb = model(zb).squeeze(1).cpu().numpy()
            out[i:i + pb.shape[0]] = pb

    return out



# Main


def main(UNet64) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    # Load data
    x = load_batches("k_set_64x64", TRAIN_BATCH_IDS)
    y = load_batches("h_set_64x64", TRAIN_BATCH_IDS)

    x_test = load_batches("k_set_64x64", TEST_BATCH_IDS)
    y_test = load_batches("h_set_64x64", TEST_BATCH_IDS)

    x = torch.tensor(x.reshape((-1, 1, 64, 64)))
    y = torch.tensor(y.reshape((-1, 64, 64)))

    x_test = torch.tensor(x_test.reshape((-1, 1, 64, 64)))
    y_test = torch.tensor(y_test.reshape((-1, 64, 64)))

    # Normalize
    z = torch.log(x) - LOGK_CENTER
    y = (y - H_MEAN) / H_STD

    z_test = torch.log(x_test) - LOGK_CENTER
    y_test = (y_test - H_MEAN) / H_STD

 
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("using device:", device)

    z = z.to(device)
    y = y.to(device)
    z_test = z_test.to(device)
    y_test = y_test.to(device)

    
    # Model

    model = UNet64(
        base_ch=BASE_CH,
        enforce_dirichlet_row0=ENFORCE_DIRICHLET_ROW0
    ).to(device)

    loss_fn = nn.MSELoss()

    optim = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    # Baseline (mean field)
    mean_field = torch.mean(y, dim=0)
    baseline = torch.mean((y_test - mean_field) ** 2).item()
    print("baseline loss:", baseline)

  
    # Checkpoints
    
    out_dir = Path("models/unet44")
    out_dir.mkdir(exist_ok=True)

    best_test = float("inf")
    best_epoch = 0
    bad_epochs = 0

    idx_all = np.arange(z.shape[0])

    
    # Training loop
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(idx_all)

        losses = []
        n_batches = (z.shape[0] - 1) // BATCH_SIZE + 1

        for b in range(n_batches):
            idx = idx_all[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

            pred = model(z[idx]).squeeze(1)
            loss = loss_fn(pred, y[idx])

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            pred_test = model(z_test).squeeze(1)
            test_loss = loss_fn(pred_test, y_test).item()

        lr_before = optim.param_groups[0]["lr"]
        scheduler.step(test_loss)
        lr_after = optim.param_groups[0]["lr"]

        if lr_after < lr_before:
            print(f"lr reduced: {lr_before:.2e} → {lr_after:.2e}")

        dt = time.time() - t0
        print(
            f"epoch {epoch:03d}/{N_EPOCHS} | "
            f"lr {lr_after:.2e} | "
            f"train {train_loss:.6f} | "
            f"test {test_loss:.6f} | "
            f"{dt:.1f}s"
        )

        # Save last
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "baseline_loss": baseline,
            "norm": {"logk_center": LOGK_CENTER, "h_mean": H_MEAN, "h_std": H_STD},
            "model": {"name": "UNet64", "base_ch": BASE_CH},
        }
        torch.save(ckpt, out_dir / "model_last.pt")

        # Best + early stopping
        if test_loss < best_test - MIN_DELTA:
            best_test = test_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(ckpt, out_dir / "model_best.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"early stopping (best epoch {best_epoch}, loss {best_test:.6f})")
            break


    # Export predictions
    best = torch.load(out_dir / "model_best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred_test = predict_in_batches(model, z_test, PRED_BATCH_TEST)

    np.savetxt(out_dir / "pred_test.txt", pred_test.reshape((-1, 4096)))

    print("saved predictions + checkpoints")
    print(f"best test loss: {best_test:.6f} (epoch {best_epoch})")


from model import Model

if __name__ == "__main__":
    main(Model)

