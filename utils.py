import numpy as np

def load_files(filename: str, file_ids: list[str]):
    return np.concatenate([np.loadtxt(filename + file_id + ".txt") for file_id in file_ids])

# IMPORTANT: We had first split our data into a train set and a validation set, with the idea of generating a validation
# set when we would need it for early stopping for example. However we accidentally started using our test set for this
# so now we use our test set for early stopping and we generated a new set of data the models have never seen during
# training to do our performance estimate. This set is called true_test

# some files might still use a hardcoded copies of these lines so be careful when modifying these but like dont change
# these because then all models need to be retrained probably.
train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000",
                  "_7000to8000"]
train_file_ids64 = ["0", "1", "2", "3", "4", "5"]
test_file_ids = ["_1000to1050", "_1050to1400"]
test_file_ids64 = ["6", "7"]
true_test_ids = ["_8000to9000", "_9000to10000"]
true_test_ids64 = ["8", "9"]

x_file_name = "datasets/k_set"
x_file_name64 = "datasets/k_set_64x64_batch"
y_file_name = "datasets/h_set"
y_file_name64 = "datasets/h_set_64x64_batch"

def load_x_train():
    """ the input (k-field)"""
    return load_files(x_file_name, train_file_ids)

def load_y_train():
    """ the output (h-field)"""
    return load_files(y_file_name, train_file_ids)

def load_x_test():
    """ the input (k-field)"""
    return load_files(x_file_name, test_file_ids)

def load_y_test():
    """ the output (h-field)"""
    return load_files(y_file_name, test_file_ids)

def load_x_true_test():
    """ the input (k-field)"""
    return load_files(x_file_name, true_test_ids)

def load_y_true_test():
    """ the output (h-field)"""
    return load_files(y_file_name, true_test_ids)

def load_x_train64():
    """ the input (k-field)"""
    return load_files(x_file_name64, train_file_ids64)

def load_y_train64():
    """ the output (h-field)"""
    return load_files(y_file_name64, train_file_ids64)

def load_x_test64():
    """ the input (k-field)"""
    return load_files(x_file_name64, test_file_ids64)

def load_y_test64():
    """ the output (h-field)"""
    return load_files(y_file_name64, test_file_ids64)

def load_x_true_test64():
    """ the input (k-field)"""
    return load_files(x_file_name64, test_file_ids64)

def load_y_true_test64():
    """ the output (h-field)"""
    return load_files(y_file_name64, test_file_ids64)
