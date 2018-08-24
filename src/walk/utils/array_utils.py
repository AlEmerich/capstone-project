import numpy

def fit_normalize(x, low, high):
    if x < 0:
        return x * (-low) # to make it positive
    return x * high
