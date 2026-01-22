# unet.py
# U-Net for 60x60 -> 60x60 regression (logK -> normalized head)

import torch
from torch import nn

train_mode = 'custom'  # unet should also work

class ConvBlock(nn.Module):
    """
    Standard double conv block.
    Keeps height/width same.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Model(nn.Module):
    """
    U-Net specifically for 60x60 grid.
    Includes the 'Global Brain' fix in the bottleneck to catch that linear gradient.
    """

    def __init__(self, base_ch: int = 64, enforce_dirichlet_row0: bool = True):
        super().__init__()

        self.enforce_dirichlet_row0 = enforce_dirichlet_row0

        # Encoder 
        # Input is 1 channel (logK map)
        in_ch = 1

        # Level 1: 60x60
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)  # drops to 30x30

        # Level 2: 30x30
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.pool2 = nn.MaxPool2d(2)  # drops to 15x15

        # Bottleneck: 15x15
        self.center = ConvBlock(2 * base_ch, 4 * base_ch)

        
        # Global Information Injection

        # Squash 15x15 spatial dims to 1x1.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        feature_ch = 4 * base_ch 

        # 2. MLP to figure out the global slope/offset
        self.global_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_ch, feature_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feature_ch, feature_ch),
            nn.Unflatten(1, (feature_ch, 1, 1))  # Reshape to (N, C, 1, 1) for broadcasting
        )

        # Decoder 

        # Level 2 Up: 15 -> 30
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(4 * base_ch + 2 * base_ch, 2 * base_ch)

        # Level 1 Up: 30 -> 60
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(2 * base_ch + base_ch, base_ch)

        # Output projection
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

        # boundary value
        self.dirichlet_row0_value = (100.0 - 145.3243) / 35.5957

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 60, 60)

        #Encoder
        x1 = self.enc1(x)  # 60x60
        x2 = self.enc2(self.pool1(x1))  # 30x30

        #Bottleneck
        x_center = self.pool2(x2)  # 15x15
        x_center = self.center(x_center)

        #Global Injection
        # 1. Get average state of the system
        global_feat = self.global_pool(x_center)
        # 2. Process through dense layer
        global_feat = self.global_dense(global_feat)
        # 3. Add back to feature map (Residual connection)
        # Broadcasting: (N, C, 15, 15) + (N, C, 1, 1)
        x_center = x_center + global_feat

        #Decoder
        d2 = self.up2(x_center)
        d2 = torch.cat([d2, x2], dim=1)  # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)  # Skip connection
        d1 = self.dec1(d1)

        out = self.out(d1)

        # Force physics on the top row if flag is set
        if self.enforce_dirichlet_row0:
            out[:, :, 0, :] = self.dirichlet_row0_value

        return out


def custom_train():
    # you need to put your custom training code here if the flag training mode is set to custom
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from torch import nn

    
    # Config
    TRAIN_FILE_IDS = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000",
                      "_7000to8000"]
    TEST_FILE_IDS = ["_1000to1050", "_1050to1400"]

    LOGK_CENTER = 4.0
    H_MEAN = 145.3243
    H_STD = 35.5957

    N_EPOCHS = 80
    BATCH_SIZE = 16

    BASE_CH = 64  # try 32, 48, 64
    LR = 3e-4  # try 1e-3, 3e-4, 1e-4
    WEIGHT_DECAY = 1e-5  # try 0, 1e-6, 1e-5, 1e-4
    ENFORCE_DIRICHLET_ROW0 = True

    # Early stopping
    PATIENCE = 12  # stop if best test_loss doesn't improve for this many epochs
    MIN_DELTA = 1e-4  # required improvement to reset patience

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4

    # Export predictions in batches to avoid OOM
    PRED_BATCH_TRAIN = 16
    PRED_BATCH_TEST = 64

    SEED = 0

    
    # Data loading 
    def load_files(prefix_path: str, file_ids: list[str]) -> np.ndarray:
        # loadtxt defaults to float64; force float32 to save memory
        parts = [np.loadtxt(prefix_path + fid + ".txt", dtype=np.float32) for fid in file_ids]
        return np.concatenate(parts, axis=0)

    def predict_in_batches(model: nn.Module, z: torch.Tensor, batch_size: int) -> np.ndarray:
        """Run model(z) in batches and return numpy float32 array (N,60,60) on CPU."""
        model.eval()
        n = z.shape[0]
        out = np.empty((n, 60, 60), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n, batch_size):
                zb = z[i:i + batch_size]
                pb = model(zb).squeeze(1).detach().float().cpu().numpy()
                out[i:i + pb.shape[0]] = pb

        return out

    def main() -> None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Load and preprocess
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

        z = z.to(device)
        y = y.to(device)
        z_test = z_test.to(device)
        y_test = y_test.to(device)


        # Model / loss / optimizer
        model = Model(base_ch=BASE_CH, enforce_dirichlet_row0=ENFORCE_DIRICHLET_ROW0).to(device)
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

        # Clean baseline: predict the mean head field (train mean) for every test sample
        mean_field = torch.mean(y, dim=0)  # (60,60)
        baseline = torch.mean((y_test - mean_field) ** 2).item()
        print("baseline loss:", baseline)


        # Checkpoints / logging
        out_dir = Path("models/unet")
        out_dir.mkdir(exist_ok=True)

        best_test = float("inf")
        best_epoch = 0
        bad_epochs = 0

        batch_idx = np.arange(z.shape[0])


        # Training loop
        for epoch in range(1, N_EPOCHS + 1):
            t0 = time.time()

            model.train()
            np.random.shuffle(batch_idx)

            epoch_losses = []
            n_batches = (z.shape[0] - 1) // BATCH_SIZE + 1

            for b in range(n_batches):
                idx = batch_idx[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]

                pred = model(z[idx]).squeeze(1)  # (B,1,60,60) -> (B,60,60)
                loss = loss_fn(pred, y[idx])

                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                epoch_losses.append(loss.item())

            train_loss = float(np.mean(epoch_losses))

            model.eval()
            with torch.no_grad():
                pred_test = model(z_test).squeeze(1)  # (N,1,60,60) -> (N,60,60)
                test_loss = loss_fn(pred_test, y_test).item()

            lr_now = optim.param_groups[0]["lr"]
            scheduler.step(test_loss)
            new_lr = optim.param_groups[0]["lr"]
            if new_lr != lr_now:
                print(f"  lr reduced: {lr_now:.2e} -> {new_lr:.2e}")

            dt = time.time() - t0
            lr_now = optim.param_groups[0]["lr"]
            print(
                f"epoch {epoch:>3d}/{N_EPOCHS} | "
                f"lr {lr_now:.2e} | train_loss {train_loss:.6f} | test_loss {test_loss:.6f} | {dt:.1f}s"
            )

            # Save "last" every epoch
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "baseline_loss": baseline,
                "norm": {"logk_center": LOGK_CENTER, "h_mean": H_MEAN, "h_std": H_STD},
                "train_file_ids": TRAIN_FILE_IDS,
                "test_file_ids": TEST_FILE_IDS,
                "model": {"name": "UNet60", "base_ch": BASE_CH, "enforce_dirichlet_row0": ENFORCE_DIRICHLET_ROW0},
                "train": {"batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED},
            }
            torch.save(ckpt, out_dir / "model_last.pt")

            # Track best + early stopping
            if test_loss < best_test - MIN_DELTA:
                best_test = test_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save(ckpt, out_dir / "model_best.pt")
            else:
                bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print(
                    f"early stopping: no improvement for {PATIENCE} epochs (best epoch {best_epoch}, best test {best_test:.6f})")
                break


        # Export predictions from best checkpoint
        best = torch.load(out_dir / "model_best.pt", map_location=device)
        model.load_state_dict(best["model_state_dict"])
        model.eval()


        if device.type == "cuda":
            torch.cuda.empty_cache()

        pred_test = predict_in_batches(model, z_test, batch_size=PRED_BATCH_TEST)

        np.savetxt("models/unet/pred_test.txt", pred_test.reshape((-1, 3600)))

        print("saved:", "models/unet/pred_test_unet.txt,", str(out_dir / "model_last.pt"),
              str(out_dir / "model_best.pt"))
        print(f"best test loss: {best_test:.6f} (epoch {best_epoch})")

    main()
    return
