import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from utils import load_files
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, padding='same')  # -> 100x100
        self.conv2 = nn.Conv2d(5, 3, 7, stride=3, padding='valid')  # -> 32x32
        self.conv3 = nn.Conv2d(3, 2, 5, stride=3, padding='valid')  # -> 10x10
        self.fc1 = nn.Linear(200, 100)  # -> 10x10
        self.conv4 = nn.Conv2d(3, 3, 5, padding='same')  # -> 32x32
        self.conv5 = nn.Conv2d(6, 3, 7, padding='same')  # -> 100x100
        self.conv6 = nn.Conv2d(8, 4, 7, padding='same')  # -> 100x100
        self.conv7 = nn.Conv2d(4, 1, 5, padding='same')  # -> 100x100
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.fc1(x3.flatten(start_dim=1))).reshape((-1, 1, 10, 10))
        x5 = self.relu(self.conv4(torch.cat((x4, 0*x3), dim=1)))
        x5 = nn.functional.interpolate(x5, size=(32, 32))
        x6 = self.relu(self.conv5(torch.cat((x5, 0*x2), dim=1)))
        x6 = nn.functional.interpolate(x6, size=(100, 100))
        x7 = self.relu(self.conv6(torch.cat((x6, 0.5*x1), dim=1)))
        y = self.conv7(x7)[:, 0, :, :]
        return y





class BorderPad_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        #first channel must be h
        super().__init__()
        self.n_pad = kernel_size//2
        self.refl_pad = nn.ReflectionPad2d(self.n_pad)

    def forward(self, x):
        z = self.refl_pad(x)
        z[:, 0, :, :self.n_pad//2] = (100-146)/37 - z[:, 0, :, :self.n_pad//2]












#test fc after conv

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
    model_id = "12c2"
    model = Model12c().to(device)
    print([i.numel() for i in model.parameters()], sum([i.numel() for i in model.parameters()]))

    # continue form existing model
    #last = torch.load(out_dir / f"model{model_id}_last.pt", map_location=device)
    #model.load_state_dict(last["model_state_dict"])

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=4e-5)

    # Baseline (predict mean field from train set)
    baseline = loss_fn(torch.mean(y, dim=0, keepdim=True), y_test).item()
    print("baseline loss:", baseline)

    # -------------------------
    # Training loop
    # -------------------------
    n_epochs = 200
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
