import numpy as np


def fit_normalize(x, low, high):
    if x < 0:
        return x * -low
    return x * high


scale = np.vectorize(fit_normalize)
