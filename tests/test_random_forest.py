import numpy as np
from ml_from_scratch.supervised.random_forest import RandomForest


def test_learns_separable_data():

    # Usual test to ensure that model learns easily separable data

    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    forest = RandomForest(n_trees=10, max_depth=3, random_state=0)
    forest.train(X, Y)

    preds = forest.predict(X)
    assert list(preds) == [0, 0, 0, 1, 1, 1]


def test_correct_number_of_trees():

    # Ensures that the requested number of trees are produced

    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    Y = np.array([0, 0, 1, 1])
    forest = RandomForest(n_trees=7, max_depth=2, random_state=0)
    forest.train(X, Y)

    assert len(forest.trees) == 7


def test_predicts_new_points():

    # Ensures new points are successfully predicted in easily separable data

    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    forest = RandomForest(n_trees=10, max_depth=3, random_state=0)
    forest.train(X, Y)

    assert forest.predict(np.array([[1.5]]))[0] == 0
    assert forest.predict(np.array([[11.5]]))[0] == 1