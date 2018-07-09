import numpy

def fit_normalize(x, low, high):
    if x < 0:
        return x * (-low)
    return x * high
