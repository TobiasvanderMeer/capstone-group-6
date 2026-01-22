import numpy as np


def badness_score(k):
    # this function tries to predict when the model will fail. If returns low values for samples that are probably not
    # going to be correctly predicted. This function takes as input a batch of conductivity fields and outputs an array
    # with one value for each value in the batch
    bordermin = np.minimum(np.min(k[:, :, :, :4], axis=(2, 3)), np.min(k[:, :, :4, :], axis=(2, 3)))
    bordermin = np.log(bordermin)
    return bordermin.reshape((-1))