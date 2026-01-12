import numpy as np
import matplotlib.pyplot as plt
from utils import load_files

set_ids = ["_1000to1050", "_1050to1400"]  # the set that was used as test set in the cnn file
#set_id = "0"
#set_id = "_test"
result_id = "_test"  # we want to look at the performance on the test set

x = load_files("datasets/k_set", set_ids).reshape((-1, 60, 60))
y = load_files("datasets/h_set", set_ids).reshape((-1, 60, 60))
pred = np.loadtxt(f"pred{result_id}.txt").reshape((-1, 60, 60))*37+146

print("MAE (full test set): ", np.mean(np.abs(y-pred)))

for i in range(100):
    print("MAE: ", np.mean(np.abs(y[i] - pred[i])))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.log(x[i].transpose()))
    # plt.ion()
    ax2.imshow(y[i], interpolation='none')
    ax2.contour(y[i], levels=20, colors=["black"])
    ax3.imshow(pred[i], interpolation='none')
    ax3.contour(pred[i], levels=20, colors=["black"])
    plt.show(block=True)
