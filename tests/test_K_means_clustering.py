import numpy as np
from ml_from_scratch.unsupervised.K_means_clustering import KMeans


def test_finds_obvious_clusters():

    # Tests that algorithm finds clusters that are easily separated

    rng = np.random.default_rng(0)
    blob1 = rng.standard_normal((20, 2)) + [0, 0]     
    blob2 = rng.standard_normal((20, 2)) + [10, 10]    
    blob3 = rng.standard_normal((20, 2)) + [0, 10]     
    X = np.vstack([blob1, blob2, blob3])

    model = KMeans(k=3, max_iters=100, random_state=42)
    model.fit(X)

    assert len(np.unique(model.labels[:20])) == 1
    assert len(np.unique(model.labels[20:40])) == 1
    assert len(np.unique(model.labels[40:60])) == 1


def test_correct_number_of_centroids():

    # Tests that model assigns the correct number of centroids

    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 2))

    model = KMeans(k=3, max_iters=50, random_state=0)
    model.fit(X)

    assert model.centroids.shape == (3, 2)


def test_predict_assigns_new_points():

    # Tests that predict method assigns labels to new points

    blob_low = np.zeros((10, 2))                  
    blob_high = np.full((10, 2), 20.0)            
    X = np.vstack([blob_low, blob_high])

    model = KMeans(k=2, max_iters=50, random_state=0)
    model.fit(X)


    new_points = np.array([[0.5, 0.5], [19.5, 19.5]])
    preds = model.predict(new_points)

    assert preds[0] != preds[1]                  
    assert preds[0] == model.labels[0]
    assert preds[1] == model.labels[15]