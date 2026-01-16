import numpy as np
import matplotlib.pyplot as plt


# Use the same test split order as cnn.py (edit if your cnn.py uses different test_file_ids)
# These match the ones from your train_unet.py run
set_ids = ["_1000to1050", "_1050to1400"]

# Which prediction file to visualize:
# CHANGED: Pointing to your new U-Net output
model_id = "fc1"
pred_file = f"models/{model_id}/pred_test.txt"

# Load and concatenate test sets in the same order
x = np.concatenate([np.loadtxt(f"datasets/k_set{sid}.txt") for sid in set_ids]).reshape((-1, 60, 60))
y = np.concatenate([np.loadtxt(f"datasets/h_set{sid}.txt") for sid in set_ids]).reshape((-1, 60, 60))

# Model outputs are normalized (h_norm). Unnormalize back to head units.
pred = np.loadtxt(pred_file).reshape((-1, 60, 60)) * 37 + 146

print("MAE (full test set): ", np.mean(np.abs(y-pred)))
# Plot at most this many samples (avoid 100 popups)
MAX_PLOTS = 10

n = min(len(x), len(y), len(pred), MAX_PLOTS)
for i in range(n):
    print("MAE: ", np.mean(np.abs(y[i] - pred[i])))
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))
    ax1.plot(pred[i, -1, :] - pred[i, -2, :])
    ax1.plot(y[i, -1, :] - y[i, -2, :])
    ax2.plot(pred[i, :, -1] - pred[i, :, -2])
    ax2.plot(y[i, :, -1] - y[i, :, -2])
    ax3.plot(pred[i, 0, :])
    ax3.plot(y[i, 0, :])
    ax4.plot(pred[i, :, 1] - pred[i, :, 0])
    ax4.plot(y[i, :, 1] - y[i, :, 0])
    ax4.plot(-50/x[i, :, 0])
    plt.legend(["pred", "truth", "theoretical"])
    plt.show()