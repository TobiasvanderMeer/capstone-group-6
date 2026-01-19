import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import importlib
from pathlib import Path

# Import model and functions
from jeffrey_code import hydraulic_conductivity_field, solve_darcy_flow, source_function

model_id = "cnn15"  # change this to compare different models
postfix = "_test"


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


# Settings #TODO add batch comparison
n = 60
MAX_SAMPLES = 10
PLOT_H = False
np.random.seed(42)
seeds = np.random.choice(range(10000), size=MAX_SAMPLES, replace=False)
f = source_function(n)

true_times = []
pred_times = []


print("starting tests")
# Loop over 10 random samples
for i, seed in enumerate(seeds):
    # Generate random hydraulic conductivity field
    kappa = hydraulic_conductivity_field(n, seed)

    # True hydraulic head
    start_true = time.time()
    h_true = solve_darcy_flow(n, kappa, f)
    end_true = time.time()
    true_times.append(end_true - start_true)

    # Predicted hydraulic head
    x_input = torch.tensor(np.log(kappa) - logk_center, dtype=torch.float32).view((60, 60)).unsqueeze(0).unsqueeze(0).to(device)
    start_pred = time.time()
    with torch.no_grad():
        h_pred_norm = model(x_input)
    end_pred = time.time()
    pred_times.append(end_pred - start_pred)
    print(f"sample {i+1}: Jeffreys code {true_times[-1]*1000:.2f}ms, machine learning model {pred_times[-1]*1000:.2f}ms")

    if PLOT_H:
        h_pred = h_pred_norm.cpu().numpy()[0] * h_std + h_mean

        # Plot True vs Predicted
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(h_true.reshape((60, 60)), cmap='hot_r', origin='lower', interpolation='none')
        axes[0].set_title(f'True h\nTime: {true_times[-1]:.4f}s')
        axes[0].set_xlabel('x (km)')
        axes[0].set_ylabel('y (km)')

        axes[1].imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
        axes[1].set_title(f'Predicted h\nTime: {pred_times[-1]:.4f}s')
        axes[1].set_xlabel('x (km)')
        axes[1].set_ylabel('y (km)')

        plt.tight_layout()
        plt.show()


# Average computation times
print(f'Average computation time for TRUE h: {np.mean(true_times):.4f}s per image')
print(f'Average computation time for PREDICTED h: {np.mean(pred_times):.4f}s per image')
