import numpy as np

def euclidian_distance (p,q):
    return np.sqrt((p-q) @ (p-q))


class KNN:

    def __init__(self, k):
        self.k = k
    
    def store_data(self, features, labels):
        self.feature_data = np.asarray(features, dtype = float)
        self.label_data = np.asarray(labels)
    
    def find_neighbors(self, point):
        neighbors = []

        for index, feature in enumerate(self.feature_data):
            dist = euclidian_distance(point, feature)
            temp_data = [feature, self.label_data[index], dist]
            neighbors.append(temp_data)
        
        neighbors.sort(key = lambda x : x[-1])

        return neighbors[:self.k]

    def predict(self, point, regression = False):
        neighbors = self.find_neighbors(point)

        if regression:
            return sum(x[1] for x in neighbors)/self.k
        else:
            labels = [x[1] for x in neighbors]
            return max(labels, key = labels.count)
    
