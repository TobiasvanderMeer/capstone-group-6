import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import torch
from pathlib import Path

# --------------------------
# Import functions and model from your code
# --------------------------
from jeffrey_code import (
    hydraulic_conductivity_field,
    solve_darcy_flow,
    hydraulic_head_gradient,
    source_function
)
from cnn import Model12c  # <-- Replace with actual filename without .py

# --------------------------
# Device (GPU if available)
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Load trained model checkpoint
# --------------------------
model_id = "12c2"
model = Model12c().to(device)
ckpt_path = Path("checkpoints") / f"model{model_id}_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()  # Set to evaluation mode

# Normalization constants
logk_center = ckpt["norm"]["logk_center"]  # typically 4.0
h_mean = ckpt["norm"]["h_mean"]
h_std = ckpt["norm"]["h_std"]

# --------------------------
# Settings
# --------------------------
n = 60
MAX_SAMPLES = 10
np.random.seed(42)
seeds = np.random.choice(range(10000), size=MAX_SAMPLES, replace=False)

f = source_function(n)

true_times = []
pred_times = []

# --------------------------
# Loop over samples
# --------------------------
for idx, seed in enumerate(seeds):
    # Generate random K field
    kappa = hydraulic_conductivity_field(n, seed)

    # True hydraulic head
    start_true = time.time()
    h_true = solve_darcy_flow(n, kappa, f)
    end_true = time.time()
    true_times.append(end_true - start_true)

    # --------------------------
    # Model prediction
    # --------------------------
    # Transform input for model
    x_input = torch.tensor(np.log(kappa) - logk_center, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    start_pred = time.time()
    with torch.no_grad():
        h_pred_norm = model(x_input)  # Output normalized
    end_pred = time.time()
    pred_times.append(end_pred - start_pred)

    # Unnormalize to original h units
    h_pred = h_pred_norm.cpu().numpy()[0] * h_std + h_mean

    # Compute gradient for streamplot from true h
    U, V = hydraulic_head_gradient(n, h_true, kappa)
    Y, X = np.mgrid[0:n:n*1j, 0:n:n*1j]
    speed = np.sqrt(U**2 + V**2)
    lw = 5 * speed / speed.max()
    lw = lw.reshape(n, n)

    # --------------------------
    # Plot log(K) and predicted h
    # --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: log(K)
    ax = axes[0]
    im = ax.imshow(np.log(kappa), cmap='jet', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title(f'Sample {seed}\nlog(K)')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')

    # Right: predicted h
    ax = axes[1]
    im = ax.imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.streamplot(X, Y, U, V, color='black', linewidth=lw, density=[0.5,1])
    ax.set_title(f'Predicted h\nTrue: {true_times[-1]:.4f}s, Pred: {pred_times[-1]:.4f}s')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')

    plt.tight_layout()
    plt.show()

# --------------------------
# Print average computation times
# --------------------------
print(f'Average computation time for TRUE h: {np.mean(true_times):.4f}s per image')
print(f'Average computation time for PREDICTED h: {np.mean(pred_times):.4f}s per image')
