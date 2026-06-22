import numpy as np
from ml_from_scratch.supervised.single_neuron import SingleNeuron, identity, sigmoid

"""

As inputting identity and sigmoid functions into the built SingleNeuron class gives us both linear and logistic regression, the linear/logistic regression tests
will live in the SingleNeuron test file

"""
#
#      Linear regression tests:
#
def test_linear_relationship():
 
# Tests a perfectly linear dataset which the model should be able to predict pretty accurately

    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)   
    y = 2 * X.flatten() + 1                

    model = SingleNeuron(activation_function=identity)
    model.train(X, y, alpha=0.01, epochs=1000)

    preds = model.predict(X)

    assert np.mean((preds - y) ** 2) < 1.0



def test_error_decreases():

# Tests that final error is less that initial error

    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.flatten() - 2
    model = SingleNeuron(activation_function=identity)
    model.train(X, y, alpha=0.01, epochs=500)

    assert model.errors_[-1] < model.errors_[0]



def test_finds_effective_line():

# Tests that the learned line has similar slope and intercept (weight 1 and bias) to the true line

    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + 1
    model = SingleNeuron(activation_function=identity)
    model.train(X, y, alpha=0.01, epochs=3000)

    assert abs(model.w_[0] - 2) < 0.5      
    assert abs(model.w_[-1] - 1) < 0.5


#
#     Logistic regression tests:
#


def test_outputs_are_probabilities():

# Tests that outputs are in between 0 and 1

    np.random.seed(0)
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])
    model = SingleNeuron(activation_function=sigmoid)
    model.train(X, y, alpha=0.1, epochs=500)

    prob = model.predict(X)
    assert np.all(prob >= 0) and np.all(prob <= 1)

def test_separates_groups():

# Tests that two very distinct groups can be separated

    np.random.seed(0)
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = SingleNeuron(activation_function=sigmoid)
    model.train(X, y, alpha=0.1, epochs=2000)

    preds = np.where(model.predict(X) >= 0.5, 1, 0)
    assert np.array_equal(preds, y)

def test_higher_input_gives_higher_probability():

# Tests that higher values will be assigned probability closer to 1 than lower values

    np.random.seed(0)
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = SingleNeuron(activation_function=sigmoid)
    model.train(X, y, alpha=0.1, epochs=2000)

    low_prob = model.predict(np.array([[0.0]]))
    high_prob = model.predict(np.array([[10.0]]))
    assert high_prob > low_prob