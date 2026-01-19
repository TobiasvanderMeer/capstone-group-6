import numpy as np

def load_files(filename: str, file_ids: list[str]):
    return np.concatenate([np.loadtxt(filename + file_id + ".txt") for file_id in file_ids])

# some files might still use a hardcoded copies of these lines so be careful when modifying these
train_file_ids = ["0", "_1400to2000", "_2000to3000", "_3000to4000", "_4000to5000", "_5000to6000", "_6000to7000",
                  "_7000to8000"]
x_file_name = "datasets/k_set"
test_file_ids = ["_1000to1050", "_1050to1400"]
y_file_name = "datasets/h_set"

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
