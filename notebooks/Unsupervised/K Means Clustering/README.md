# K Means Clustering

K means clustering is our only unsupervised algorithm. This means that it organizes data solely based on the features. Given an integer k, this algorithm attempts to separate a dataset into k distinct blobs by predicting where the center of those blobs should be. It does this by first picking k random centers, then labeling the points that are close to each centroid. Once it has done this, it then reassigns centroids to be in the middle of each group (denoted by which centroid they were previously closest to) and then iterates until either the centroid doesn't really move or a certain number of trial has been reached.

---

## The Actual Math

The algorith aims to partition the data inputted into $k$ separate clusters so that the points within a cluster are as close as possible, effectively separating the data into a couple "blobs." The algorithm does this with an iterative process as follows: First, it picks $k$ random points in the dataset and denotes them to be "centroids." We will call these points $c_1, ... , c_k.$ It then organizes the other data into $k$ sets $C_1, ..., C_k$ such that for a dataset $D,$

$$C_n = \{x \in D: d(x, c_n) > d(x, c_j) \forall 1 \leq j \leq k, j \neq n\}$$

Note $d(x,y)$ here is the euclidean distance function we talked about in KNN. Put more plainly, the algorithm labels each point bases on which centroid it is closest to. After that, the algorithm iterates through $C_1, ..., C_k$ and averages out every feature creating a new centroid so

$$c_n' = \frac{1}{|C_n|}\sum_{x \in C_n}x$$

it does this for each $c_n$ creating $c_1', ... ,c_k'$ and then does the same process on these newly chosen centroids. Either after a certain number of iterations, or after the average distance the centroids move is below a certain threshold, the algorithm stops.

## Example on Real Data

We will demonstrate this on the iris dataset with the labels removed, the goal is to see whether K-means can accurately predict the iris data which we can test for accuracy by comparing the predicted versus real.