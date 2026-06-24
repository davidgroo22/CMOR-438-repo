import numpy as np
from ml_from_scratch.supervised.decision_tree import DecisionTree


def test_gini_pure_node_is_zero():

    # Ensures that gini_index calculates a uniform set correctly

    tree = DecisionTree()
    pure = np.array([1, 1, 1, 1])
    assert np.isclose(tree.gini_index(pure), 0.0)

def test_learns_separable_data():

    # Tests that the tree effectively learns easily separable data

    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    tree = DecisionTree(min_samples_split=2, max_depth=3)
    tree.train(X, Y)

    preds = tree.predict(X)
    assert list(preds) == [0, 0, 0, 1, 1, 1]


def test_predicts_correct_class_on_new_point():

    # Tests that the predict method accurately predicts an easily classifiable point

    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    tree = DecisionTree(max_depth=3)
    tree.train(X, Y)

    assert tree.predict(np.array([[1.5]]))[0] == 0
    assert tree.predict(np.array([[11.5]]))[0] == 1