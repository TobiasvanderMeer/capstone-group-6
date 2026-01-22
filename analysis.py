import numpy as np
import matplotlib.pyplot as plt
import utils
import torch

# Which prediction file to visualize (change these two):
model_id = "cnn_fc64"
postfix = "_b50"  # use "" for no postfix

n = 64 if model_id in ["unet88", "unet44"] else 60  # choose the right resolution

# Plot at most this many samples (avoid 100 popups)
MAX_PLOTS = 10
# show the samples in order of descending error (worst predictions first)
SHOW_ORDERED_BY_ERROR = False
RESHOW_CONVERGENCE = True  # this show the convergence plot again (same as the one saved, but useful if you want to zoom in)

#this will be the file we will need to analise (don't change this line)
pred_file = f"models/{model_id}/pred_test{postfix}.txt"

if RESHOW_CONVERGENCE:
    # this show the convergence plot again
    last = torch.load(f"models/{model_id}/model_last{postfix}.pt")
    train_losses = last["train_loss_history"]
    test_losses = last["test_loss_history"]
    total_train_time = last["training_time"]
    print(f"this model was trained in {total_train_time} seconds")
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_loss", "test_loss"])
    print("showing convergence")
    plt.show()

# Load test sets
if n == 60:
    x = utils.load_x_test().reshape((-1, n, n))
    y = utils.load_y_test().reshape((-1, n, n))
elif n == 64:
    x = utils.load_x_test64().reshape((-1, n, n))
    y = utils.load_y_test64().reshape((-1, n, n))
    if not model_id == "unet44":
        x = x[:1000]
        y = y[:1000]
else:
    raise ValueError(f"{n} is not a supported resolution")

# Model outputs are normalized (h_norm). Unnormalize back to head units.
if model_id in ["unet", "unet88", "unet44"]:
    H_MEAN = 145.3243  # 146
    H_STD = 35.5957  # 37
else:
    H_MEAN = 146
    H_STD = 37

pred = np.loadtxt(pred_file).reshape((-1, n, n)) * H_STD + H_MEAN

print("MAE (full test set): ", np.mean(np.abs(y-pred)))

n = min(len(x), len(y), len(pred), MAX_PLOTS)

if SHOW_ORDERED_BY_ERROR:
    # show the samples in order of decreasing MAE (useful for investigating outliers)
    MAEs = np.mean(np.abs(y-pred), axis=(1, 2))
    #plt.plot(MAEs, "o")
    #plt.show()
    order = [list(MAEs).index(i) for i in sorted(list(MAEs), reverse=True)]
else:
    # this show the samples in order of the dataset
    order = range(n)


for i in order:
    print("MAE: ", np.mean(np.abs(y[i] - pred[i])))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))

    # Calculate min/max from the TRUTH so both plots use the same scale
    # This helps you compare them fairly
    vmin, vmax = np.min(y[i]), np.max(y[i])

    # make plots
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
