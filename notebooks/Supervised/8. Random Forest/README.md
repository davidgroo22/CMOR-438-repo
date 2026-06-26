# Random Forest

The random forest algorithm is basically a bunch of decision trees trained on parts of the data and then used as a voting system to predict the class of a certain object. As discussed in the decision tree section, it is easy to overfit with decision trees, so this method allows for a large amount of data analysis by the model without necessitating a large depth value. Even though the models may have some amount of error in a certain direction, with enough of them the error will cancel out causing this to be an incredibly effective algorithm.

---

# The Actual Math

A random forest utilizes a technique called "bootstrap aggregation" or bagging. Say we want to train $n$ trees which we will denote as functions $T_1(x), ..., T_n(x)$. Call $D$ to be our set of data then we will pick $k$ random points from $D$ with replacement to train each of our trees on. Each tree will see a different collection of the same data, allowing for example $T_1$ to pick up on a good split in the fourth feature while $T_4$ picks up a good split on the 20th feature. Once we have trained all of these trees using the decision tree algorithm in the previous folder, we have created our random forest. To predict a data point $x_0$ we pass $x_0$ into all of the trees and take

$$\hat{y} = \text{mode}(T_1(x_0), ... , T_n(x_0))$$

to be our prediction. When one tree makes an strong error in one direction, its vote only counts for 1 so as long as few other trees made the same error, the aggregate vote will accurately predict the class of the data point.

## Example on Real Data

We will now test thi data on the breast cancer dataset that we tested the decision tree on. We will be able to then compare the results of the random forest and the decision tree which may give us insight on the benefits of one algorithm versus the other. We will do this in the jupyter notebook in this folder.