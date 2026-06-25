# K Nearest Neighbors

The K Nearest Neighbors algorithm is likely the simplest of the machine learning algorithms outlined in this repository. KNN can be used as a regression or classification method. The algorithm simply given a point, iterates through all of the data to find a certain number of the nearest points. If you are using regression, it averages the value you are attempting to predict as a prediction for the point you are looking at. If you are using classification, it simply takes the most common class between the k nearest neighbors.

## The Actual Math

The math behind KNN is incredibly simple. The only real math behind KNN is the Euclidean distance function which given $x = (x_1, ... , x_n)$ and $y = (y_1, ... , y_n)$ we have that the Euclidean distance 

$$d(x, y) = \sqrt{\sum_{1}^n ({x_i} - {y_i})^2}$$

Given a dataset $\vec{x_1}, ... , \vec{x_j}$ with labels $y_1, ... y_j$ given a point $\vec{x_i}$ we iterate through the rest of the data and find the $k$ points with the smallest Euclidean distance from $x_i.$ Call these points $\vec{x_{n_1}}, ... , \vec{x_{n_k}}.$ The rest of the process depends on if we are using regression or classification. If we're using regression, then we take the average of $y_{n_1}, ... , y_{n_k}$ and use that value to predict $y_i.$ If we are using classification, then we take the most common value between $y_{n_1}, ... , y_{n_k}$ and use that as the prediction for $y_i.$ Note that this algorithm is what we call a lazy learning algorithm in that it doesn't actually learn anything, it just predicts and memorizes data.

---

## Example on real data

We will now show the KNN method on the same iris dataset that was used on the perceptron. A benefit of using this method is that we are able to classify multiple classes of data rather than just one. Because of this, we will be able to use the full iris dataset rather than just 2 of the classes.
