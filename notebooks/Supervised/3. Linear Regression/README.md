# Linear Regression

Linear Regression is a machine learning technique that is used to effectively attempt to draw a line through a set of data that minimizes the overall distance from all the points. It does this by utilizing the previous algorithm, gradient descent, to optimize a set of weights (which act as slopes) for the line. Over enough iterations, it is able to find the "line of best fit" through the data. Note that linear regression is not a classification algorithm but rather a regression algorithm, therefore it will attempt to predict a numerical value rather than a class.

---

## The Actual Math

A lot of the math for the linear regression algorithm we have actually already seen in the previous two algorithms. Similarly to the Perceptron, linear regression begins with randomized weights $w_1, ... w_k$ and a bias $b.$ Given data points $(x_{1,1}, x_{1, 2}, ... , x_{1, k}, y_1), ... , (x_{n, 1}, x_{n, 2}, ... , x_{n, k}, y_n).$ Given $\vec{x_i}$ the model will then output

$$\hat{y_i} = w_1x_{i, 1} + ... + w_kx_{i, k} + b$$

as its prediction. Note that this is similar to the perceptron except in that case it is filtered by the sign function whereas here it is the raw value. Once the model has done this, it uses gradient descent to attempt to minimize the cost function or move the line closer to the point that was passed through. From the same cost function $C(\vec{w}) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y_i} - y_i)^2$ we get

$$\frac{\partial C}{ \partial \vec{w}} = \frac{1}{n}\sum_{i=1}^{n}{\vec{x_i}}(\hat{y_i} - y)$$ and $$\frac{\partial C}{ \partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y_i} - y)$$

so then as per gradient descent, we shift the weights as described by gradient descent such that:

$$w \leftarrow w - \eta \frac{\partial C}{\partial w}$$ and $$b \leftarrow b - \eta \frac{\partial C}{\partial b}$$

Additionally my implementation in specific does this stochastically so each update is based off of exactly one point. It then repeats this for as many iterations as it is told to and as shown earlier when discussing gradient descent, it will tend towards the minimum cost function. Unlike classification algorithms like perception, accuracy of linear regression is calculated through "Mean Squared Error" which is simply

$$\frac{1}{2}\sum_{i=1}^n(\hat{y_i}-y_i)^2$$

but note that this is almost exactly the cost function that we are minimizing. Because of this, the whole effect of gradient descent is effectively minimizing how "wrong" the model is which in turn, tends to it being pretty accurate after enough time.
