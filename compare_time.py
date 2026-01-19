import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import torch
from pathlib import Path


# Import functions and model from your code
from jeffrey_code import (
    hydraulic_conductivity_field,
    solve_darcy_flow,
    hydraulic_head_gradient,
    source_function
)
from cnn import Model12c


# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained model checkpoint
model_id = "12c2"
model = Model12c().to(device)
ckpt_path = Path("checkpoints") / f"model{model_id}_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()  # Set to evaluation mode

logk_center = ckpt["norm"]["logk_center"] 
h_mean = ckpt["norm"]["h_mean"]
h_std = ckpt["norm"]["h_std"]


# Settings
n = 60
MAX_SAMPLES = 10
np.random.seed(42)
seeds = np.random.choice(range(10000), size=MAX_SAMPLES, replace=False)

f = source_function(n)

true_times = []
pred_times = []



# Loop over random samples and plot
for idx, seed in enumerate(seeds):
    # Generate random hydraulic conductivity field
    kappa = hydraulic_conductivity_field(n, seed)

    # Compute TRUE hydraulic head
    start_true = time.time()
    h_true = solve_darcy_flow(n, kappa, f)
    end_true = time.time()
    true_times.append(end_true - start_true)

    # Run CNN model prediction
    x_input = torch.tensor(np.log(kappa) - logk_center, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    start_pred = time.time()
    with torch.no_grad():
        h_pred_norm = model(x_input)  # model output normalized
    end_pred = time.time()
    pred_times.append(end_pred - start_pred)

    # Unnormalize back to original units
    h_pred = h_pred_norm.cpu().numpy()[0] * h_std + h_mean

    # Plot TRUE and PREDICTED h side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # True hydraulic head
    ax = axes[0]
    im = ax.imshow(h_true, cmap='hot_r', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'True h\nTime: {true_times[-1]:.4f}s')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')

    # Predicted hydraulic head
    ax = axes[1]
    im = ax.imshow(h_pred, cmap='hot_r', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Predicted h\nTime: {pred_times[-1]:.4f}s')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')

    plt.tight_layout()
    plt.show()


# Print average computation times
print(f'Average computation time for TRUE h: {np.mean(true_times):.4f}s per image')
print(f'Average computation time for PREDICTED h: {np.mean(pred_times):.4f}s per image')
