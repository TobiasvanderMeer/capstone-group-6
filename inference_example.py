import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import importlib
from pathlib import Path

# Import model and functions
from jeffrey_code import hydraulic_conductivity_field, solve_darcy_flow, source_function


model_id = "cnn12c"  # change this to use different models
postfix = "_lr6e-5"

model_file = importlib.import_module(f"models.{model_id}.model")  # this line imports the right model and training settings

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Checkpoint folder
out_dir = Path("models") / model_id
out_dir.mkdir(exist_ok=True)

# Load model checkpoint
ckpt_path = out_dir / f"model_last{postfix}.pt"
ckpt = torch.load(ckpt_path, map_location=device)

model = model_file.Model().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

logk_center = ckpt["norm"]["logk_center"]
h_mean = ckpt["norm"]["h_mean"]
h_std = ckpt["norm"]["h_std"]


# Settings
n = 60
BATCH_SIZE = 16
PLOT_H = True
np.random.seed(42)
seeds = np.random.choice(range(10000), size=BATCH_SIZE, replace=False)
f = source_function(n)


# Generate random hydraulic conductivity field; replace this with the input
kappa = np.empty((BATCH_SIZE, 3600))
for j, seed in enumerate(seeds):
    kappa[j] = hydraulic_conductivity_field(n, seed)

# Predicted hydraulic head
x_input = torch.tensor(np.log(kappa) - logk_center, dtype=torch.float32).view((-1, 1, 60, 60)).to(device)

with torch.no_grad():
    h_pred_norm = model(x_input)

if PLOT_H:
    h_pred = h_pred_norm.cpu().numpy()[0] * h_std + h_mean

    plt.imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
    plt.title(f'Predicted h')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')

    plt.tight_layout()
    plt.show()


