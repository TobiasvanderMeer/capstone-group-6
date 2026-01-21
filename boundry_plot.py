import numpy as np
import matplotlib.pyplot as plt
import utils

# Which prediction file to visualize:
model_id = "unet88"
postfix = ""
pred_file = f"models/{model_id}/pred_test{postfix}.txt"
n = 64

# Load test sets
if n == 60:
    x = utils.load_x_test().reshape((-1, n, n))
    y = utils.load_y_test().reshape((-1, n, n))
elif n == 64:
    x = utils.load_x_test64().reshape((-1, n, n))
    y = utils.load_y_test64().reshape((-1, n, n))
else:
    raise ValueError(f"{n} is not a supported resolution")

# Model outputs are normalized (h_norm). Unnormalize back to head units.
pred = np.loadtxt(pred_file).reshape((-1, n, n)) * 37 + 146

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