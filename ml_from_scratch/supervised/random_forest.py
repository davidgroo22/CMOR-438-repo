from ml_from_scratch.supervised.decision_tree import DecisionTree
import numpy as np
import pandas as pd
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, Y, rng):
        
        # Picks random row indices and returns them

        n = X.shape[0]
        indices = rng.choice(n, size=n, replace=True)
        return X[indices], Y[indices]

    def train(self, X, Y):

        # Trains n trees utilizing previously implemented decision tree class

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y).reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for _ in range(self.n_trees):
            X_sample, Y_sample = self._bootstrap_sample(X, Y, rng)
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth)
            tree.train(X_sample, Y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):

        # Predicts utilizing random forest method, aggregates predictions from each tree

        X = np.asarray(X, dtype=float)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_preds.append(most_common)
        return np.array(final_preds)