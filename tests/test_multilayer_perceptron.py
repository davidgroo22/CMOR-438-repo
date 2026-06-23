import numpy as np
from ml_from_scratch.supervised.multilayer_perceptron import NeuralNet


def test_forward_output_shape():
    # 5 images in -> output should be (5, 10): 5 rows of 10 digit-probabilities
    X = np.random.rand(5, 784)
    net = NeuralNet(random_state=0)
    output = net.forwardprop(X)

    assert output.shape == (5, 10)


def test_forward_rows_sum_to_one():
    # Softmax output: each row (each image's 10 probabilities) must sum to 1
    X = np.random.rand(5, 784)
    net = NeuralNet(random_state=0)
    output = net.forwardprop(X)

    row_sums = output.sum(axis=1)
    # every row sum should be very close to 1.0
    assert np.allclose(row_sums, 1.0)


def test_forward_outputs_are_probabilities():
    # All outputs must be valid probabilities: between 0 and 1
    X = np.random.rand(5, 784)
    net = NeuralNet(random_state=0)
    output = net.forwardprop(X)

    assert np.all(output >= 0) and np.all(output <= 1)


def test_intermediates_are_stored():
    # Backprop needs the cached intermediate values. Confirm forwardprop
    # stores them on self with the correct shapes.
    X = np.random.rand(5, 784)
    net = NeuralNet(random_state=0)