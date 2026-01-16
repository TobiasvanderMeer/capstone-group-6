import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils import load_files
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    """
    Standard double conv block.
    Keeps height/width same.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

class UNet2(nn.Module):

    #doesn't train

    def __init__(self, base_ch: int = 8, enforce_dirichlet_row0: bool = False):
        super().__init__()

        self.enforce_dirichlet_row0 = enforce_dirichlet_row0

        # Encoder path
        # -----------------------
        # Input is 1 channel (logK map)
        in_ch = 1

        # Level 1: 60x60
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(3)  # drops to 20x20

        # Level 2: 30x30
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.pool2 = nn.MaxPool2d(2)  # drops to 10x10

        #level 3:
        self.enc3 = ConvBlock(2*base_ch, 4*base_ch)
        self.pool3 = nn.MaxPool2d(2)  # drops to 5x5

        # Bottleneck: 5x5
        self.center = CenterBlock(4 * base_ch, 8 * base_ch)

        # Decoder path
        # -----------------------
        # Level 3 Up: 5 -> 10
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(8 * base_ch + 4 * base_ch, 4 * base_ch)

        # Level 2 Up: 10 -> 20
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(4 * base_ch + 2 * base_ch, 2 * base_ch)

        # Level 1 Up: 20 -> 60
        self.up1 = nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(2 * base_ch + base_ch, base_ch)

        # Output projection
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Hardcoded boundary value for top row (physically h=100m, normalized)
        # y = (h - 146) / 37
        self.dirichlet_row0_value = (100.0 - 145.3243) / 35.5957

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 60, 60)

        # --- Encoder ---
        x1 = self.enc1(x)  # 60x60
        x2 = self.enc2(self.pool1(x1))  # 20x20
        x3 = self.enc3(self.pool2(x2))

        # --- Bottleneck ---
        x_center = self.pool3(x3)  # 5x5
        x_center = self.center(x_center)


        # --- Decoder ---
        d3 = self.up3(x_center)
        d3 = torch.cat([d3, x3], dim=1)  # Skip connection
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
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



if __name__ == "__main__":
    train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000", "_7000to8000"]
    #train_file_ids = ["0"]
    test_file_ids = ["_1000to1050", "_1050to1400"]

    x = torch.tensor(load_files("datasets/k_set", train_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z = torch.log(x)-4
    y = (torch.tensor(load_files("datasets/h_set", train_file_ids).reshape((-1, 60, 60)), dtype=torch.float)-146) / 37

    x_test = torch.tensor(load_files("datasets/k_set", test_file_ids).reshape((-1, 1, 60, 60)), dtype=torch.float)
    z_test = torch.log(x_test) - 4
    y_test = (torch.tensor(load_files("datasets/h_set", test_file_ids).reshape((-1, 60, 60)), dtype=torch.float) - 146) / 37


    print(torch.mean((y - torch.mean(y, dim=0))**2))
    print(torch.mean(y))

    # -------------------------
    # Device (GPU if available)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # Move full datasets to device once (fast + simple; fits easily in VRAM here)
    z = z.to(device)
    y = y.to(device)
    z_test = z_test.to(device)
    y_test = y_test.to(device)

    # Checkpoint folder
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)

    # -------------------------
    # Model / loss / optimizer
    # -------------------------
    model_id = "u_net2"
    model = UNet2().to(device)
    print([i.numel() for i in model.parameters()], sum([i.numel() for i in model.parameters()]))

    # continue form existing model
    #last = torch.load(out_dir / f"model{model_id}_last.pt", map_location=device)
    #model.load_state_dict(last["model_state_dict"])

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=8e-4)

    # Baseline (predict mean field from train set)
    baseline = loss_fn(torch.mean(y, dim=0, keepdim=True), y_test).item()
    print("baseline loss:", baseline)

    # -------------------------
    # Training loop
    # -------------------------
    n_epochs = 12
    batch_size = 16
    batch_idx = np.arange(z.shape[0])

    best_test = float("inf")

    train_losses = []
    test_losses = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        np.random.shuffle(batch_idx)
        epoch_losses = []

        n_batches = (z.shape[0] - 1) // batch_size + 1
        for i in range(n_batches):
            idx = batch_idx[i * batch_size:(i + 1) * batch_size]

            pred = model(z[idx])
            loss = loss_fn(pred, y[idx])

            loss.backward()
            optim.step()
            optim.zero_grad()

            #print(epoch, i, loss.item())
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            pred_test = model(z_test)
            test_loss = loss_fn(pred_test, y_test).item()

        dt = time.time() - t0
        print(f"epoch {epoch}/{n_epochs} | train_loss {train_loss:.6f} | test_loss {test_loss:.6f} | {dt:.1f}s")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # -------------------------
        # Save checkpoints (last + best)
        # -------------------------
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "baseline_loss": baseline,
            # Normalization constants used in this code:
            "norm": {"logk_center": 4.0, "h_mean": 145.3243, "h_std": 35.5957},
            # Helpful metadata:
            "train_file_ids": train_file_ids,
            "test_file_ids": test_file_ids,
        }

        torch.save(ckpt, out_dir / f"model{model_id}_last.pt")
        if test_loss < best_test:
            best_test = test_loss
            torch.save(ckpt, out_dir / f"model{model_id}_best.pt")

    # -------------------------
    # Save predictions
    # -------------------------
    best = torch.load(out_dir / f"model{model_id}_best.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])

    model.eval()
    with torch.no_grad():
        pred_test = model(z_test).detach().cpu().numpy()

    np.savetxt(f"pred_test{model_id}.txt", pred_test.reshape((-1, 3600)))

    print("saved:", f"pred_train{model_id}.txt, pred_test{model_id}.txt, checkpoints/model{model_id}_last.pt, checkpoints/model{model_id}_best.pt")

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.savefig(f"convergence_plot{model_id}.png")
    plt.show()
