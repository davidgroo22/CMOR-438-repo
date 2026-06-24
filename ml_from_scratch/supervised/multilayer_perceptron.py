import numpy as np

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def softmax(Z):

    # Shifts final weights to probabilities that add up to 1
    # This method was suggested by Claude

    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp = np.exp(Z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

class NeuralNet:
    def __init__(self, input_layer = 784, hidden_layer1 = 16, hidden_layer2 = 16, output_layer = 10, random_state = None):
        rng = np.random.default_rng(random_state)

        # Input layer to first hidden layer weights set to random and biases set to 0

        self.W1 = rng.standard_normal((input_layer, hidden_layer1)) * 0.1
        self.b1 = np.zeros(hidden_layer1)

        # First hidden layer to second hidden layer. weights set to random and biases set to 0

        self.W2 = rng.standard_normal((hidden_layer1, hidden_layer2)) * 0.1
        self.b2 = np.zeros(hidden_layer2)

        # Second hidden layer to output layer weights set to random and biases set to 0

        self.W3 = rng.standard_normal((hidden_layer2, output_layer)) * 0.1
        self.b3 = np.zeros(output_layer)

    def forwardprop(self, X):

        # Ensures X is of the type we want

        X = np.asarray(X, dtype=float)
        self.X = X

        # First layer --> Second layer
        # Matrix dims: (n x 784) * (784 x 16) + (16, ) = (n x 16)

        self.Z1 =  X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)

        # Second layer --> Third layer
        # Matrix dims: (n x 16) * (16 x 16) + (16, ) = (n x 16)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)

        # Third layer --> Output
        # Matrix dims: (n x 16) * (16 x 10) + (10, ) = (n x 10)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.Out_Matrix = softmax(self.Z3)

        # Returns (n x 10) matrix of probabilities that sum to 1

        return self.Out_Matrix
    
    def predict(self, X):

        # Uses forwardprop to predict X

        probs = self.forwardprop(X)
        return np.argmax(probs, axis = 1)


    def backprop(self, Y, alpha):

        # Utilizes the backpropogation method as displayed in the 3Blue1Brown video

        n = Y.shape[0]

        delta3 = self.Out_Matrix - Y                                      
        delta2 = (delta3 @ self.W3.T) * (self.A2 * (1 - self.A2))  # (n, 16)
        delta1 = (delta2 @ self.W2.T) * (self.A1 * (1 - self.A1))  # (n, 16)

        dW3 = self.A2.T @ delta3 / n      # (16, 10)
        db3 = np.sum(delta3, axis=0) / n  # (10,)
        dW2 = self.A1.T @ delta2 / n      # (16, 16)
        db2 = np.sum(delta2, axis=0) / n  # (16,)
        dW1 = self.X.T @ delta1 / n       # (784, 16)
        db1 = np.sum(delta1, axis=0) / n  # (16,)

        self.W3 -= alpha * dW3
        self.b3 -= alpha * db3
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1

    def train(self, X, Y, alpha=0.5, epochs=100):
        
        # Trains the model using the method outlined in the 3Blue1Brown video

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.costs = []

        for _ in range(epochs):
            self.forwardprop(X)                                  
            cost = -np.sum(Y * np.log(self.Out_Matrix + 1e-9)) / X.shape[0]  
            self.costs.append(cost)
            self.backprop(Y, alpha)                                
        return self



        