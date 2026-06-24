# Perceptron

The Perceptron is effectively a vastly simplified model of a Neuron that you would find in the brain. It receives a certain amount of inputs, that all work together to determine an output. Then, the Perceptron learns based off of how similar the value it outputs was to the true value. It adjusts how it percieves each input in relation to the output, and overtime is able to learn the correct output given specific inputs.

---

## The Actual Math

Consider a dataset of data points of the form $(x_1, x_2, ... , x_k, l)$ where $l$ is a label and $x_1, ... , x_k$ are numerical values describing each data point. The Perceptron model has $k+1$ stored values that are randomized at first, $w_1, w_2, ... , w_k,$ and $b$ which are $k$ weight values and a bias value. Given a data point of the previous form, the Perceptron does two things. First, it calculates a raw value $z$ which is the weighted sum of the inputs such that
$$z = w_1x_1 + w_2x_2 + ... + w_kx_k + b$$
Then, the Perceptron turns the raw value into an actual prediction using the sign function which simply takes the sign of the output and we call that prediction $\hat{y}$ such that
$$\hat{y} = sign(z) = sign(w_1x_1 + w_2x_2 + ... + w_kx_k + b)$$
The value of $\hat{y}$ is the prediction that is given by the Perceptron, however it is often wrong when untrained. In order to actually train our model, we must update the weights so that the model is more likely to predict correctly next time. This is how the Perceptron "Learns."