from ml_from_scratch.supervised.perceptron import Perceptron
import numpy as np

def test_learns_separable_data():
    # Easy separation to test classification
    X = np.array([
        [0,0], [0,1], [1, 0], [1,1], 
        [6,7], [6,6], [7,7], [7,6]
                 ])
    y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    model = Perceptron()
    model.train(X,y)

    preds = model.predict(X)

    assert np.array_equal(preds, y)

def test_predict_returns_valid_labels():
    # Tests that the outputs are only -1 or 1
    X = np.array([[0, 0], [6, 7], [1, 1], [6, 6]])
    y = np.array([-1, 1, -1, 1])
    model = Perceptron(epochs=20)
    model.train(X, y)

    preds = model.predict(X)
  
    assert set(np.unique(preds)).issubset({-1, 1})

def test_errors_recorded_each_epoch():
    # Ensures that errors are recorded during each epoch
    X = np.array([[0, 0], [6, 7], [1, 1], [6, 6]])
    y = np.array([-1, 1, -1, 1])
    epochs = 15
    model = Perceptron(epochs=epochs)
    model.train(X, y)

    assert len(model.errors) == epochs