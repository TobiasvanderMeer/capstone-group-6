import numpy as np

def load_files(filename: str, file_ids: list[str]):
    return np.concatenate([np.loadtxt(filename + file_id + ".txt") for file_id in file_ids])
