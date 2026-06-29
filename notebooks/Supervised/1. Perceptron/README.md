# Perceptron

The Perceptron is effectively a vastly simplified model of a Neuron that you would find in the brain. It receives a certain amount of inputs, that all work together to determine an output. Then, the Perceptron learns based off of how similar the value it outputs was to the true value. It adjusts how it percieves each input in relation to the output, and overtime is able to learn the correct output given specific inputs.

---

## The Actual Math

Consider a dataset of data points of the form $(x_1, x_2, ... , x_k, l)$ where $l$ is a label and $x_1, ... , x_k$ are numerical values describing each data point. The Perceptron model has $k+1$ stored values that are randomized at first, $w_1, w_2, ... , w_k,$ and $b$ which are $k$ weight values and a bias value. Given a data point of the previous form, the Perceptron does two things. First, it calculates a raw value $z$ which is the weighted sum of the inputs such that

$$z = w_1x_1 + w_2x_2 + ... + w_kx_k + b$$

Then, the Perceptron turns the raw value into an actual prediction using the sign function which simply takes the sign of the output and we call that prediction $\hat{y}$ such that

$$\hat{y} = sign(z) = sign(w_1x_1 + w_2x_2 + ... + w_kx_k + b)$$

The value of $\hat{y}$ is the prediction that is given by the Perceptron, however it is often wrong when untrained. In order to actually train our model, we must update the weights so that the model is more likely to predict correctly next time. This is how the Perceptron "Learns." The rule we utilize to do this is as follows

$$\vec{w} \leftarrow \vec{w} + \eta (l - \hat{y})\vec{x}$$ and $$b \leftarrow b + \eta(l - \hat{y})$$

where $\vec{w}$ is the weight vector, $\eta$ is a chosen learning rate (we mostly used 0.5 in this class), $\vec{x}$ is the vector of the $x_i$ and $l$ is the correct value assigned to the label of the data point. Note that the closer that $l$ is to $\hat{y},$ the less the model needs to update itself. Because of this, with enough reasonable data, the model will tend towards being more and more accurate. The algorithm however determines a line separating the data, so if the data is not "linearly separable" that is if the data cannot be separated into classes by a line, the perceptron will never become 100% accurate.

---

## A Note about Convergence of the Perceptron

One issue with the perceptron is that it can only achieve 100% accuracy on data that is "linearly separable" or data in $\mathbb{R}^n$ such that we can draw a line in $\mathbb{R}^n$ that separates the classes. We will consider the proof of the **Perceptron Convergence Theorem** to show that:

The perceptron converges to 100% accuracy $\iff$ the data it is trained on is linearly separable.

Call $D = \{(\vec{x_i}, y_i)\}_{i=1}^n$ to be our dataset with $y_i = \pm 1$ for each $1 \leq i \leq n.$ Additionally we will say that $D$ is **linearly separable** if there exists a weight vector $\vec{w}'$ and bias $b'$ such that $y_i(\vec{w}' \dot (\vec{x_i},1)) > 0$ for each $x_i \in D.$ Additionally we will say that the perceptron converges to 100% accuracy if there is $k$ such that the $k$th weight and bias vectors, $\vec{w_k}, b_k$ satisfies $(\vec{w_k} \dot (\vec{x_i}, 1))y_i \geq 0$ for each i.

First notice by these definitions that if the perceptron converges to 100% accuracy on step $k$ then we have that $D$ is linearly separable with $\vec{w}', b' = \vec{w_k}, b_k$ so we have thus shown that direction

Now we suppose that the data is linearly separable and we are then given some $\vec{w}'$ that separates the data. Additionally we can normalize this vector such that $||\vec{w}'|| = 1$ (The Euclidean Norm). Additionally as $y_i(\vec{w}' \dot (\vec{x_i},1)) > 0$ we have $B$ such that

$$y_i(\vec{w}' \dot (\vec{x_i},1)) > B$$

for each of our finite datapoints. Call $\vec{w_k}$ the weight vector of our perceptron after $k$ updates then note that for some $x_i \in D$ we have that

$$\vec{w_k} = \vec{w_{k-1}} + x_iy_i$$

Additionally then dotting both sides we get

$$\vec{w_k} \dot \vec{w}' = \vec{w_{k-1}} \dot \vec{w}' + B = \vec{w_{k-2}} \dot \vec{w'} + 2B = kB.$$

Additionally then applying Cauchy-Schwarz we have $\vec{w_k} \dot \vec{w}' \leq ||\vec{w_k}|| || \vec{w}'|| = ||\vec{w_k}||$ thus we get

$$kB \leq ||\vec{w_k}||$$

Now, as $D$ is finite, take $R: ||\vec{x_i}|| \leq R$ for each $i.$ Note also that 

$$||\vec{w_k}||^2 = ||\vec{w_{k-1}} + y_i\vec{x_i}||^2 = ||\vec{w_{k-1}}||^2 + 2y_i(\vec{w_{k-1} \dot \vec{x_i}}) + ||x_i||^2$$

Note also that $2y_i(\vec{w_{k-1} \dot \vec{x_i}}) < 0$ because it is the twice the update step from a misclassified point, i.e. the algorithm wouldn't be updating if it were positive. Additionally as $||\vec{x_i}||^2 \leq R^2$ we then get that

$$||\vec{w_k}||^2 \leq ||\vec{w_{k-1}}||^2 + R^2 \leq ||\vec{w_{k-2}}||^2 + 2R^2 \leq ... \leq kR^2$$

From this, we have that 

$$k^2B^2 \leq ||\vec{w_k}||^2 \leq kR^2 \implies k \leq \frac{R^2}{B^2}$$

giving us that $k$ is bounded above. Because of this, the algorithm can only make $\lfloor \frac{R^2}{B^2} \rfloor$ updates before it converges, thus it converges in a finite number of steps. From this we have shown both directions and now can conclude that

The perceptron converges to 100% accuracy $\iff$ the data it is trained on is linearly separable.

---

## Examples on real Data

Now in the Jupyter notebook included in this package, I will demonstrate this algorithm on two classifications in the iris dataset imported from Scikit-learn.