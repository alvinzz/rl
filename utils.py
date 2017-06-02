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

def run_model(model, inputs, stochastic=True):
    num_layers = int(len(model) / 2)
    h = relu(np.matmul(inputs, model["W0"]) + model["b0"])
    for i in 1 + np.arange(num_layers - 2, dtype=np.int32):
        h = relu(np.matmul(h, model["W{}".format(i)]) + model["b{}".format(i)])
    probs = sigmoid(np.matmul(h, model["W{}".format(num_layers - 1)]) + model["b{}".format(num_layers - 1)])
    if stochastic:
        actions = len(probs)
        return np.random.choice(np.arange(len(probs)), p=probs)
    else:
        return np.argmax(probs)
