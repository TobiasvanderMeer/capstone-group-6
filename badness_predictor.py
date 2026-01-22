import torch
import numpy as np


def badness_score(k):
    bordermin = np.minimum(np.min(k[:, :, :, :4], axis=(2, 3)), np.min(k[:, :, :4, :], axis=(2, 3)))
    bordermin = np.log(bordermin)
    return bordermin.reshape((-1))