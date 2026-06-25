# Logistic Regression

Logistic Regression is a machine learning technique that utilizes gradient descent but as a classification method rather than regression. It is referred to as regression because it assigns continuous values to data but those values represent probabilities. In specific, it assigns data points probabilities of being in one of two classes. These probabilities can then be used to help classify points. It does this in a similar way to linear regression, but passes a function in called the sigmoid instead of the identity. The sigmoid function jumps very quickly from being near 0 to being near 1 which allows a separation of the data with some values in the middle that are assigned probabilities closer to 0.5. I then utilize a classifier function that separates points with probabilities above and below 0.5 in order to classify points.

---

## The Actual Math

Similar to the previous two algorithms, logistic regression begins with random weights $w_1, ... ,w_k$ and a bias $b$ such that given a datapoint $(x_1, x_2, ..., x_k, y)$ such that $y \in \{0, 1\}$ it computes the sum:

$$z = w_1x_1 + ... + w_kx_k + b$$

once it does this, instead of passing it as a raw value like linear regression, it instead applies a function onto z called the sigmoid function which is defined so

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

This produces $\hat{y} = \sigma(z)$ or our prediction value. This value will be in between 0 and 1 and we interpret this value as a probability. More specifically, it is a probability that the datapoint belongs to a specific class that is denoted by $y.$ If $\hat{y} > 0.5$ we predict that $y =1$ and otherwise we predict that $y = 0.$ To train this model, we use a cost function called the binary cross-entropy cost function. This is denoted as

$$C = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

Part of the reason that this cost function is so powerful is the fact that when paired with the sigmoid derivative, when we take the gradient we get

$$\frac{\partial C}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_i$$

which is the exact same gradient that we got with linear regression. This works out very nicely, and we are then able to update the weights and biases in the exact same way:

$$w \leftarrow w - \eta \frac{\partial C}{\partial w}$$ and $$b \leftarrow b - \eta \frac{\partial C}{\partial b}$$ 

with the bias partial derivative being the same as the weight but getting rid of the $x_i$. We then iterate until the model learns to a high enough degree of accuracy. A note about this, I only wrote one class for both of these algorithms because the learning actually utilizes the same math. The only difference is that I pass in a sigmoid function into my single neuron class rather than the identity. 

---

## Example on Real Data

I will now demonstrate this algorithm on the breast cancer dataset from scikit-learn which contains 30 medical features and attempts to classify tumors as malignant or benign. I will test this in the jupyter notebook contained in this folder.