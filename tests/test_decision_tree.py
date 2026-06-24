import numpy as np
from ml_from_scratch.supervised.decision_tree import DecisionTree


def test_gini_pure_node_is_zero():
    # A node with all the same label is "pure" -> Gini should be 0.
    tree = DecisionTree()
    pure = np.array([1, 1, 1, 1])
    assert np.isclose(tree.gini_index(pure), 0.0)


def test_gini_even_split_is_half():
    # A perfect 50/50 two-class split has Gini = 1 - (0.5^2 + 0.5^2) = 0.5
    tree = DecisionTree()
    even = np.array([0, 0, 1, 1])
    assert np.isclose(tree.gini_index(even), 0.5)


def test_learns_separable_data():
    # Two cleanly separable classes: low values -> 0, high values -> 1.
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    tree = DecisionTree(min_samples_split=2, max_depth=3)
    tree.train(X, Y)

    preds = tree.predict(X)
    assert list(preds) == [0, 0, 0, 1, 1, 1]


def test_predicts_correct_class_on_new_point():
    # After training on separable data, a new low point should predict 0,
    # a new high point should predict 1.
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    Y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    tree = DecisionTree(max_depth=3)
    tree.train(X, Y)

    assert tree.predict(np.array([[1.5]]))[0] == 0
    assert tree.predict(np.array([[11.5]]))[0] == 1


def test_two_feature_classification():
    # Classes separable on the first feature; tree should still get it right.
    X = np.array([[0, 0], [1, 1], [0, 1], [8, 8], [9, 9], [8, 9]])
    Y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)
    tree = DecisionTree(max_depth=3)
    tree.train(X, Y)

    preds = tree.predict(X)
    assert list(preds) == [0, 0, 0, 1, 1, 1]