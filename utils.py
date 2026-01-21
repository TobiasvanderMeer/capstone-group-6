import numpy as np

def load_files(filename: str, file_ids: list[str]):
    return np.concatenate([np.loadtxt(filename + file_id + ".txt") for file_id in file_ids])

# some files might still use a hardcoded copies of these lines so be careful when modifying these but like dont change
# these because then all models need to be retrained probably.
train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000",
                  "_7000to8000"]
train_file_ids64 = ["0", "1", "2", "3", "4", "5"]
x_file_name = "datasets/k_set"
x_file_name64 = "datasets/k_set_64x64_batch"
test_file_ids = ["_1000to1050", "_1050to1400"]
test_file_ids64 = ["6"]
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
