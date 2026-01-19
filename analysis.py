import numpy as np
import matplotlib.pyplot as plt
import utils

# Which prediction file to visualize:
model_id = "cnn12c"
postfix = "_lr6e-5"  # use "" for no postfix

#this will be the file we will need to analise
pred_file = f"models/{model_id}/pred_test{postfix}.txt"

# Plot at most this many samples (avoid 100 popups)
MAX_PLOTS = 10
# show the samples in order of descending error (worst predictions first)
SHOW_ORDERED_BY_ERROR = False

# Load and concatenate test sets
x = utils.load_x_test().reshape((-1, 60, 60))
y = utils.load_y_test().reshape((-1, 60, 60))

# Model outputs are normalized (h_norm). Unnormalize back to head units.
pred = np.loadtxt(pred_file).reshape((-1, 60, 60)) * 37 + 146

print("MAE (full test set): ", np.mean(np.abs(y-pred)))

n = min(len(x), len(y), len(pred), MAX_PLOTS)

if SHOW_ORDERED_BY_ERROR:
    # show the samples in order of decreasing MAE
    MAEs = np.mean(np.abs(y-pred), axis=(1, 2))
    #plt.plot(MAEs, "o")
    #plt.show()
    order = [list(MAEs).index(i) for i in sorted(list(MAEs), reverse=True)]
else:
    order = range(n)


for i in order:
    print("MAE: ", np.mean(np.abs(y[i] - pred[i])))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))

    # Calculate min/max from the TRUTH so both plots use the same scale
    # This helps you compare them fairly
    vmin, vmax = np.min(y[i]), np.max(y[i])

    #todo: add color bar
    ax1.set_title("log(K)")
    k_plot = ax1.imshow(np.log(x[i]), origin="lower")
    f.colorbar(k_plot, ax=ax1, shrink=0.6)

    ax2.set_title("true h")
    # Added vmin/vmax to lock scales
    true = ax2.imshow(y[i], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
    ax2.contour(y[i], levels=20, colors=["black"], linewidths=0.7)
    f.colorbar(true, ax=ax2, shrink=0.6)

    ax3.set_title("pred h")
    # Added vmin/vmax to lock scales
    pred_plot = ax3.imshow(pred[i], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
    ax3.contour(pred[i], levels=20, colors=["black"], linewidths=0.7)

    ax4.set_title("pred - true")
    diff = ax4.imshow(pred[i] - y[i], interpolation='none', origin="lower")
    #ax4.imshow(np.mean(pred, axis=0) - np.mean(y, axis=0), interpolation='none')

    f.colorbar(diff, ax=ax4, shrink=0.6)
    plt.tight_layout()
    plt.show(block=True)
