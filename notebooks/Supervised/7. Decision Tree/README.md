# Decision Tree

The Decision tree algorithm is a classification algorithm that optimizes a series of binary decisions to split data in the best possible way. It trains on data by algorithmically determining what the best splits are and forms a binary tree with which it is able to predict future data. For a data point that it would like to predict, it starts at the root node and passes it down the tree where it will depending on the branch go right or left. There are two types of nodes on a decision tree, leaf and decision nodes, the data traverses down the nodes until it hits a leaf node at which the algorithm makes a prediction given by the leaf node.

---

## The Actual Math

The main optimization that this algorithm has to do is decide what value of which feature it makes sense to split on. The goal is to choose splits that separate the classes most efficiently, and we do this by attempting to minimize a value we call Gini impurity which is defined as follows:

$$\text{Gini} = 1 - \sum_{i=1}^{c} p_i^2$$

Where $p_i$ is the proportion of the group belonging to the $ith$ class and $c$ is the total number of classes. Note that if we have 100% 1 class then $\text{Gini} = 1- 1^2 = 0$ so the minimal gini score is when a node is completely uniform. To optimize the value/feature we split on, we attempt to maximize the information gain which is simply how much the split reduces impurity. For a parent node split into left and right children, we calculate information gain as:

$$G = \text{Gini}_{\text{parent}} - \left(w_L \cdot \text{Gini}_{\text{left}} + w_R \cdot \text{Gini}_{\text{right}}\right)$$

where $w_L, w_R$ are the fractions of data points in the parents that get sorted into the left and right child nodes respectively. At each node, the algorithm attempts every feature and threshold and then chooses the split with the highest information gain. If the gini of a node ever reaches 0, the algorithm creates a "leaf node" which tells the algorithm to choose a specific class. This can also happen if the node is at the maximum depth (that is preset by the user) or has too few datapoints to split. After this, the algorithm recursively builds the tree, building all of the left nodes first and then slowly moving to the right until every route down the tree ends up at a leaf node. We are then able to predict new data by passing it through the tree.

## Example on Real Data

We will once again utilize the Breast Cancer datasert from scikit-learn to test this algorithm. We will not need to scale it as we did with logistic regression because decision trees will only consider one feature at a time therefore every feature is considered equally, simply on what the most optimal information gain is. We will also attempt multiple different depths of trees to see how they effect performance.