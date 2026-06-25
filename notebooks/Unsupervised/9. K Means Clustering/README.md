# K Means Clustering

K means clustering is our only unsupervised algorithm. This means that it organizes data solely based on the features. Given an integer k, this algorithm attempts to separate a dataset into k distinct blobs by predicting where the center of those blobs should be. It does this by first picking k random centers, then labeling the points that are close to each centroid. Once it has done this, it then reassigns centroids to be in the middle of each group (denoted by which centroid they were previously closest to) and then iterates until either the centroid doesn't really move or a certain number of trial has been reached.

---

## The Actual Math

