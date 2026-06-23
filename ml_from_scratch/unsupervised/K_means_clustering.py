import numpy as np

def euclidian_distance (p,q):
    return np.sqrt((p-q) @ (p-q))

class KMeans:
    def __init__(self, k = 3, max_iters = 100, random_state = None):
        self.k = k
        self.iters = max_iters
        self.random_state = random_state

    def random_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        random_indices = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]
    
    def assign_clusters(self, X):
        labels = []
        for point in X:
            distances = [euclidian_distance(point, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)
    
    def fit(self, X):
        X = np.asarray(X, dtype = float)
        self.random_centroids(X)

        for iter in range(self.iters):
            labels = self.assign_clusters(X)

            # Reassigns centroids

            new_centroids = []
            for cluster in range(self.k):
                points_in_cluster = X[labels == cluster]
                if len(points_in_cluster) > 0:
                    new_centroids.append(points_in_cluster.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[cluster])
            self.centroids = np.array(new_centroids)

        self.labels = self.assign_clusters(X)
        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.assign_clusters(X)
    

        

