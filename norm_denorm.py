import json

import numpy as np

from utils import load_data_phase


def calculate_mean_disp(folder, params):
    x_train, y_train = load_data_phase(folder, params, (4, 4))
    mean = y_train.mean(axis=0)
    dispersion = ((y_train - mean) ** 2).mean(axis=0)
    return mean, dispersion


def norm(array, mean, dispersion):
    return (array - mean) / np.sqrt(dispersion)


def denorm(array, mean, dispersion):
    return array * np.sqrt(dispersion) + mean


def dump_mean_disp(file, mean, dispersion):
    d = {
        'mean': list(mean),
        'dispersion': list(dispersion)
    }

    with open(file, 'w') as f:
        json.dump(d, f)


def load_mean_disp(file):
    with open(file, 'r') as f:
        return map(np.array, json.load(f).values())
