import numpy as np


def show_npy_data(dataset_path):
    print(np.load(file=dataset_path, allow_pickle=True))


def load_npy_data(dataset_path):
    return np.load(file=dataset_path, allow_pickle=True)
