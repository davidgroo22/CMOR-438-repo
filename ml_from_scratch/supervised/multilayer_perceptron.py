import numpy as np
from single_neuron import sigmoid

def softmax(Z):

    # Shifts final weights to probabilities that add up to 1

    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp = np.exp(Z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

class NeuralNet:
    def __init__(self, input_layer = 784, hidden_layer1 = 16, hidden_layer2 = 16, output_layer = 10):
        rng = np.random.default_rng(random_state)

        # Input layer to first hidden layer weights set to random and biases set to 0

        self.W1 = rng.standard_normal((input_layer, hidden_layer1)) * 0.01
        self.b1 = np.zeros(hidden_layer1)

        # First hidden layer to second hidden layer. weights set to random and biases set to 0

        self.W2 = rng.standard_normal((hidden_layer1, hidden_layer2)) * 0.01
        self.b2 = np.zeros(hidden_layer2)

        # Second hidden layer to output layer weights set to random and biases set to 0

        self.W3 = rng.standard_normal((hidden_layer2, output_layer)) * 0.01
        self.b3 = np.zeros(output_layer)

    def forwardprop(self, X):

        # Ensures X is of the type we want

        X = np.asarray(X, dtype=float)



        