# Gradient Descent

Gradient Descent is the only algorithm included in this repository that is purely mathematical. That is, while the technique is utilized in certain machine learning algorithms, it is simply an optimization technique that will be relied on heavily in the next three machine learning techniques - linear and logistic regression, as well as the multilayer perceptron. When a model is attempting to accurately predict a series of numerical values, you can measure how close it was by measuring how close each output was to its expected. Gradient descent attempts to minimize the function that describes this which we call the cost function. Gradient descent uses calculus to do so which will be described below.

---

## The Actual Math

The gradient of an n-dimensional function $\nabla f$ is a vector of all of its partial derivatives in each coordinate. The gradient of a function looked at as a vector in $\mathbb{R}^n$ points in the direction in which the function's value increases the most. Similarly, the negative gradient points in the direction in which the function's value decreases the most. We will utilize this property of the negative gradient to minimize what we will call the cost function. I will explain the 2-dimensional case as it is easier to understand, however this can be generalized to any finite number of dimensions. Suppose we have a dataset $(x_1, y_1), (x_2, y_2), ... (x_n, y_n).$ Suppose that given $x_i$ we have a model that predicts $\hat{y_i}$ then we call our cost function, $C(w)$ so that

$$C(w) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y_i} - y_i)^2$$

Note also that $\hat{y_i} = wx_i + b$ so taking the gradient (or in this case the derivative of C with respect to w), we get that

$$\frac{\partial C}{ \partial w} = \frac{1}{n}\sum_{i=1}^{n}{x_i}(\hat{y_i} - y)$$

Now once we calculate this, we will take the "Steepest step downwhill" in an attempt to minimize the cost function. We will do this by replacing

$$w \leftarrow w - \eta \frac{\partial C}{\partial w}$$

where $\eta$ is a chosen learning rate (usually around 0.1) so that the algorithm doesn't overshoot but still converges at a reasonable pace. We iterate this a number of times and eventually are able to minimize the cost function giving us our desired weight. Note that this can be generalized to higher dimensions by replacing the $x_i$ and $w$ with vectors, the math remains the same.

---

## Stochastic Gradient Descent

This method is very powerful, however with large datasets or large models (foreshadowing the multilayer perceptron) it can take a very long time to sum all of these terms up. Because of this we utilize something called Stochastic Gradient Descent which instead of using every training example at once, we use one or possible a couple at a time to speed up the process. This is not quite as accurate however with enough data it will still converge to the same result.