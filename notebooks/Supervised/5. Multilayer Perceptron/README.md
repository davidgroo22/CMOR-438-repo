# The Multilayer Perceptron

The Multilayer Perceptron is likely the most mathematically complex and generally the most powerful machine learning algorithms documented in this repository. The idea behind the multilayer perceptron is the following: Remember the normal perceptron has a certain amount of weights that it will apply to data, and then with those weights it outputs a value. The Multilayer Perceptron is similar except the value that it outputs is used as a weight for another perceptron. It is built layer by layer such that a layer of perceptrons turns a large dataset into a smaller one that will be passed onto the next layer of perceptrons. It uses these weighted sums of weighted sums to predict a specifc (often much more complex) dataset. Additionally, to train the model, we need to utilize techniques much more complicated than the perceptron update rule, as the weights in layer one effect the output values in layer n. Because of this we use a calculus-based technique called backpropogation to effectively update weights after training which will be explained below.

---

## The Actual Math

First, like any model, we must go forward through the network to see what it predicts. The model will receive input $X$ which is an $n \times m$ matrix such that $m$ is the number of features we are training our model on and $n$ is the number of datapoints we are passing into our model. For our example we will consider a four layer model with arbitrary layer sizes $l_1, l_2,$ and $l_3.$ (Note we have already denoted $m$) to be the size of the input layer. Now we will have a weight matrix which we will call $W_1$ with dimensions $m \times l_1.$ Additionally we will have a bias vector $B_1$ which we will be a row vector added to each column of our matrix as such:

$$Z_1 = (X \times W_1) + B_1$$

which will calculate a matrix that will be passed into the next layer. Now recall the sigmoid function from logistic regression, we will use this function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to scale our inputs to layer 2 into probabilities as such:

$$A_1 = \sigma(Z_1)$$ 

Once we have done this, we pass $A_2$ into the next layer and repeat the exact same process but with a new set of weights and a smaller dimension. We repeat until we get to the output layer, where we calculate $Z_3$ (As we are looking at a model with four layers) and apply a new function called softmax to it such that

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

This scales $Z_1$ so each row adds up to 1, giving an accurate representation of the probabilities of each class in the output layer. We then choose the class based on the highest probability.

Now that we have effectively predicted the class of the data point, here is where the training step comes in. First recall the notion of the cost function, similarly to past models, we want to minimize this. The cost function that we use for the multilayer perceptron is

$$C = -\sum_i Y_i \log(\hat{Y_i})$$

Note that each $Y_i$ is equal to zero except for one which is equal to 1 which we will call $Y_0$. Note that then this sum is really equal to $-Y_0\log(\hat{Y_0}) = -\log(\hat{Y_0}).$ Note also that this approaches zero as $Y_0$ approaches 1 and otherwise is a large positive value as the logarithm of a number smaller than one is negative and goes towards $-\infty$. Additionally if we are inputting $n$ datapoints, we do the same thing and scale by a factor of $\frac{1}{n}$ to effectively average the cost.

Now that we have propogated forward and defined our cost function, we must reupdate the weights to step towards minimizing the cost function. Remember however as stated earlier, weights in early layers effect outputs in later layers so we cannot simply update as we have with past methods. Instead, we must use a calculus based method called backpropogation which utilizes the chain rule to effectively minimize the cost, working backwards from the output. We denote $\delta$ to be the error of each layer and we calculate $\delta$ for each layer beginning at the output layer (Note that we have 3 layers of weights $1 \rightarrow 2, 2 \rightarrow 3, 3 \rightarrow 4$) so we need three values of $\delta.$ Note that because softmax utilizes an exponential and our cost function utilizes a log, taking the derivative outputs a nice value of

$$\delta_3 = \hat{Y} - Y$$

Where $\hat{Y}$ are the prediction values and $Y$ are the real values. Now we are able to calculate the previous deltas with the formula

$$\delta_n = \left(\delta_{n+1} {W_{(n+1)}}^{T}\right) \odot \sigma'(Z_{n})$$

Note that $W_n, Z_n$ are defined above and the $\odot$ denotes elementwise multiplication retaining the shape of the matrix. Now once we are able to calculate $\delta_1, \delta_2, $ and $\delta_3$ we are then able to calculate the gradient of the cost of a certain layer. We scale this by the error to effectively fix the layers that are the most wrong the quickest. We do this by taking

$$\frac{\partial C}{\partial W_{n}} = \frac{1}{k} {A_{n-1}}^{T} \delta_{n}$$

as well as taking the gradiant of the bias to be

$$\frac{\partial C}{\partial B_{n}} = \frac{1}{k}\sum_{datapoints} \delta_{n}$$

Once we have found the gradients of each layer, we are able to update the weights and biases of each layer similarly to how we do in a single perceptron. We pick a learning rate $\eta$ to scale our steps and then we apply the following changes to each set of weights and biases:

$$W_n \leftarrow W_n - \eta \frac{\partial C}{\partial W_n}$$

$$B_n \leftarrow B_n - \eta \frac{\partial C}{\partial B_n}$$

Once we have done this, we have effectively decreased the cost function as we have subtracted the negative gradient or "stepped in the steepest downhill direction." That sums up the entire learning step, so now we iterate through sets of data in order to train our model. We repeat this process for a chosen number of epochs until the model is sufficiently trained.

---

## Testing on real data

This model is much more powerful than a lot of the past models and because of this, we will use a much more complex dataset. The dataset that we will demonstrate this model on is the MNIST dataset which renders handwritten numbers 1-10 as 28x28 grids as heatmaps with each pixels value ranging from 0-1 denoting how "white" the pixel is. We will attempt to teach our model to classify these handwritten numbers even though some of the numbers are hard for even a human to classify. This will be done in the jupyter notebook in this file