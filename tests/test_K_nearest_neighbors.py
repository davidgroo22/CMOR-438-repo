import numpy as np
from ml_from_scratch.supervised.K_nearest_neighbors import KNN, euclidian_distance


def test_classifies_obvious_point():

    # Tests that it classifies simple points

    X = np.array([[0, 0], [1, 1], [0, 1], [10, 10], [11, 11], [10, 11]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = KNN(k=3)
    model.store_data(X, y)

    pred = model.predict(np.array([0.5, 0.5]))  
    assert pred == 0

def test_euclidean_distance():

    # Tests Euclidean distance function

    p = np.array([0.0, 0.0])
    q = np.array([3.0, 4.0])
    assert np.isclose(euclidian_distance(p, q), 5.0)

def test_find_neighbors_returns_k_neighbors():
    
    # Tests that it returns the correct amount of neighbors

    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 1, 1, 1])
    model = KNN(k=3)
    model.store_data(X, y)

    neighbors = model.find_neighbors(np.array([0, 0]))
    assert len(neighbors) == 3
