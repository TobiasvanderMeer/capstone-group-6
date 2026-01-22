import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from inference import Predictor
import utils
import badness_predictor

BATCH_SIZE = 50
PLOT_RESULTS = True  # show each sample prediction in a plot
# this shows the samples in order of decreasing error. usefull for investigating outliers. only does something if above is true
SHOW_ORDERED_BY_ERROR = False
SHOW_BADNESS_SCORE_PLOT = False  # only does something if above is true
TEST_ON_GPU = True
SHOW_MODEL_INFO = True  # show a convergence plot

# what models to check. format: [(model_id, postfix, input_size)] input_size should be 60 or 64
#models = [("fc1", "", 60), ("cnn_fc", "_lr2e-5", 60), ("cnn12c", "_lr6e-5", 60), ("cnn16", "_test", 60), ("unet", "", 64),
#          ("unet88", "", 64), ("unet44", "", 64), ("unet44_noglob", "", 64), ("cnn_fc64", "_b16", 64)]
models = [("unet44_noglob", "", 64), ("cnn_fc64", "_b16", 64)]

print("loading data")
true_test_x = torch.tensor(utils.load_x_true_test().reshape((-1, 1, 60, 60)), dtype=torch.float)
true_test_y = torch.tensor(utils.load_y_true_test().reshape((-1, 60, 60)), dtype=torch.float)

true_test_x64 = torch.tensor(utils.load_x_true_test64().reshape((-1, 1, 64, 64)), dtype=torch.float)
true_test_y64 = torch.tensor(utils.load_y_true_test64().reshape((-1, 64, 64)), dtype=torch.float)

for i, (model_id, postfix, n) in enumerate(models):
    print("model: ", model_id, postfix)

    if n == 60:
        x_set = true_test_x
        y_set = true_test_y
    else:
        x_set = true_test_x64
        y_set = true_test_y64

    x_set_np = x_set.detach().numpy()
    y_set_np = y_set.detach().numpy()

    prediction = np.empty(y_set.shape)

    setup_time = time.time()
    predictor = Predictor(model_id, postfix, n, use_gpu=TEST_ON_GPU)

    print(f"number of trainable parameters {sum([i.numel() for i in predictor.model.parameters()])}")
    ckpt = predictor.ckpt
    print(f"this model was trained in {ckpt["training_time"] if "training_time" in ckpt else "Unknown"}s")
    if SHOW_MODEL_INFO:
        print("showing model convergence. Turn off SHOW_MODEL_INFO to hide this")
        if "train_loss_history" in ckpt:
            plt.title(f"convergence_plot {model_id}")
            plt.plot(ckpt["train_loss_history"])
            plt.plot(ckpt["test_loss_history"])
            plt.legend(["train_loss", "test_loss"])
            plt.show()
        else:
            print("no convergence plot available for this model")


    start_time = time.time()
    for j in range((x_set.shape[0] - 1)//BATCH_SIZE + 1):
        prediction[j*BATCH_SIZE:(j+1)*BATCH_SIZE] = predictor.predict(x_set[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
        if j == 0:
            first_batch_time = time.time()
    end_time = time.time()
    MAE = np.mean(np.abs(prediction-y_set_np))
    print("MAE of all samples:", MAE)
    print(f"total time: {end_time-setup_time:.4f}s, "
          f"avg time per batch: {(end_time - start_time)/x_set.shape[0]*1000*BATCH_SIZE:.2f}ms, "
          f"setup: {start_time - setup_time:.3f}s, "
          f"first batch {(first_batch_time-start_time)*1000}ms\n")


    # everything below is just plotting


    if PLOT_RESULTS:
        if SHOW_ORDERED_BY_ERROR:
            # show the samples in order of decreasing MAE (useful for investigating outliers)
            MAEs = np.mean(np.abs(y_set_np - prediction), axis=(1, 2))
            if SHOW_BADNESS_SCORE_PLOT:
                badnesses = badness_predictor.badness_score(x_set_np)
                badness_threshold = 0
                print(f"{len(np.where(badnesses < badness_threshold)[0])} samples are dropped by the threshold")
                print("MAE with threshold", np.mean(MAEs[np.where(badnesses > badness_threshold)]))

                # this plot shows the log of the MAE on the y-axis and the log minimum value of the conductivity near
                # the border on the x-axis
                plt.scatter(badnesses, np.log(MAEs), 5)
                plt.ylabel("log(MAE)")
                plt.xlabel("log(min(k_at_border)")
                plt.show()

            order = [list(MAEs).index(i) for i in sorted(list(MAEs), reverse=True)]
        else:
            # this show the samples in order of the dataset
            order = range(x_set.shape[0])

        for j in order:
            print("MAE: ", np.mean(np.abs(prediction[j]-y_set_np[j])))
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4))

            # Calculate min/max from the TRUTH so both plots use the same scale
            # This helps you compare them fairly
            vmin, vmax = np.min(y_set_np[j]), np.max(y_set_np[j])

            # make plots
            ax1.set_title("log(K)")
            k_plot = ax1.imshow(np.log(x_set[j, 0]), origin="lower")
            f.colorbar(k_plot, ax=ax1, shrink=0.6)

            ax2.set_title("true h")
            # Added vmin/vmax to lock scales
            true = ax2.imshow(y_set[j], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
            ax2.contour(y_set[j], levels=20, colors=["black"], linewidths=0.7)
            f.colorbar(true, ax=ax2, shrink=0.6)

            ax3.set_title("pred h")
            # Added vmin/vmax to lock scales
            pred_plot = ax3.imshow(prediction[j], interpolation="none", origin="lower", vmin=vmin, vmax=vmax)
            ax3.contour(prediction[j], levels=20, colors=["black"], linewidths=0.7)

            ax4.set_title("pred - true")
            diff = ax4.imshow(prediction[j] - y_set_np[j], interpolation='none', origin="lower")
            # ax4.imshow(np.mean(pred, axis=0) - np.mean(y, axis=0), interpolation='none')

            f.colorbar(diff, ax=ax4, shrink=0.6)
            plt.tight_layout()
            plt.show(block=True)




