import numpy as np

# how close to the border a sample should be for it to be considered close to the border
# we found this actually has very little effect on the results
BORDER_SIZE = 20

def badness_score(k):
    # this function tries to predict when the model will fail. If returns low values for samples that are probably not
    # going to be correctly predicted. This function takes as input a batch of conductivity fields and outputs an array
    # with one value for each value in the batch
    bordermin = np.minimum(np.min(k[:, :, :, :BORDER_SIZE], axis=(2, 3)), np.min(k[:, :, :BORDER_SIZE, :], axis=(2, 3)))
    bordermin = np.log(bordermin)
    return bordermin.reshape((-1))