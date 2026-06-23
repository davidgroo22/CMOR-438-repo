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

def _make_tiny_dataset(n_per_class=8, seed=0):
    """Helper: tiny, learnable dataset. 3 classes, clearly separated clusters
    in 784-dim space so a working net can memorize them quickly."""
    rng = np.random.default_rng(seed)
    X_list, label_list = [], []
    for cls in range(3):
        # each class clustered around a different random center
        center = rng.standard_normal(784) * 3
        X_list.append(center + rng.standard_normal((n_per_class, 784)) * 0.1)
        label_list.append(np.full(n_per_class, cls))
    X = np.vstack(X_list)
    labels = np.concatenate(label_list)
    Y = np.zeros((len(labels), 10))
    Y[np.arange(len(labels)), labels] = 1   # one-hot
    return X, Y, labels


def test_cost_decreases_during_training():
    # The whole point of gradient descent: cost should drop over epochs.
    X, Y, labels = _make_tiny_dataset()
    net = NeuralNet(random_state=0)
    net.train(X, Y, alpha=0.5, epochs=200)

    assert net.costs[-1] < net.costs[0]


def test_cost_history_length_matches_epochs():
    # One cost recorded per epoch.
    X, Y, labels = _make_tiny_dataset()
    epochs = 50
    net = NeuralNet(random_state=0)
    net.train(X, Y, alpha=0.5, epochs=epochs)

    assert len(net.costs) == epochs


def test_learns_separable_data():
    # On clearly separated clusters, a working net should reach high accuracy.
    X, Y, labels = _make_tiny_dataset()
    net = NeuralNet(random_state=0)
    net.train(X, Y, alpha=0.5, epochs=2000)

    accuracy = np.mean(net.predict(X) == labels)
    assert accuracy > 0.9


def test_weights_actually_change():
    # Training must modify the weights. If backprop did nothing (e.g. all
    # gradients zero), the weights would be unchanged -> bug.
    X, Y, labels = _make_tiny_dataset()
    net = NeuralNet(random_state=0)
    W1_before = net.W1.copy()      # copy! otherwise it's the same array reference

    net.train(X, Y, alpha=0.5, epochs=20)

    assert not np.allclose(net.W1, W1_before)


def test_cost_is_finite():
    # No NaN/inf during training (catches blow-ups from a too-large alpha
    # or a log(0) that slipped past the 1e-9 guard).
    X, Y, labels = _make_tiny_dataset()
    net = NeuralNet(random_state=0)
    net.train(X, Y, alpha=0.5, epochs=100)

    assert np.all(np.isfinite(net.costs))


def test_overfits_tiny_dataset():
    # A network with this much capacity SHOULD be able to perfectly memorize
    # a handful of well-separated points. This is the strongest "it learns" check.
    X, Y, labels = _make_tiny_dataset(n_per_class=4)
    net = NeuralNet(random_state=0)
    net.train(X, Y, alpha=0.5, epochs=1000)

    accuracy = np.mean(net.predict(X) == labels)
    assert accuracy == 1.0