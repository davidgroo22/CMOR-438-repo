import numpy as np

def identity(x):
    return x

def sigmoid(x):
    return (1/(1+np.exp(-x)))

class SingleNeuron():
    def __init__(self, activation_function):

        # Activation function is meant to be either the indentity or sigmoid for linear/logistic regression

        self.activation_function = activation_function
    
    def train(self, X, y, alpha = 0.005, epochs = 50):

        # Trains the SingleNeuron model using gradient descent

        self.w_ = np.random.rand(1+X.shape[1])
        self.errors_ = []
        N = X.shape[0]

        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                self.w_[:-1] -= alpha * (self.predict(xi) - target) * xi
                self.w_[-1] -= alpha * (self.predict(xi) - target)
                errors += .5*((self.predict(xi) -target) **2)
            self.errors_.append(errors/N)
        return self
    
    def predict(self, X):

        # Predicts a new classification based on old ones

        preactivation = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(preactivation)
    
    def predict_class(self, X, threshold=0.5):
        return np.where(self.predict(X) >= threshold, 1, 0)