import numpy as np
import matplotlib.pyplot as plt


# Use the same test split order as cnn.py (edit if your cnn.py uses different test_file_ids)
# These match the ones from your train_unet.py run
set_ids = ["_1000to1050", "_1050to1400"]

# Which prediction file to visualize:
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
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))

    # Calculate min/max from the TRUTH so both plots use the same scale
    # This helps you compare them fairly
    vmin, vmax = np.min(y[i]), np.max(y[i])

    ax1.set_title("log(K)")
    ax1.imshow(np.log(x[i]), origin="lower")

    ax2.set_title("true h")
    # Added vmin/vmax to lock scales
    ax2.imshow(y[i], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
    ax2.contour(y[i], levels=20, colors=["black"], linewidths=0.7)

    ax3.set_title("pred h")
    # Added vmin/vmax to lock scales
    ax3.imshow(pred[i], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
    ax3.contour(pred[i], levels=20, colors=["black"], linewidths=0.7)

    ax4.imshow(pred[i] - y[i], interpolation='none', origin="lower")
    #ax4.imshow(np.mean(pred, axis=0) - np.mean(y, axis=0), interpolation='none')

    plt.tight_layout()
    plt.show(block=True)
