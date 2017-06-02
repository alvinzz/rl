import numpy as np

def input_to_int(input):
    return int(str().join(filter(str.isdigit, input)))

def relu(x):
    return x if x > 0 else 0

relu = np.vectorize(relu)

def sigmoid(x):
    x = np.exp(x)
    x /= np.sum(x)
    return x
