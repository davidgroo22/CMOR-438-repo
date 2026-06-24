# CMOR 438 — Data Science and Machine Learning

**Author:** David Groothuis

**Course:** Rice University - CMOR 438, Summer 2026, Dr. Randy Davila

---

## Overview

This repository contains from scratch implementations of a variety of machine learning algorithms built using Python and NumPy, alongside Jupyter notebooks that demonstrate these algorithms on real datasets. Every algorithm is implemented as a reusable class in the `ml_from_scratch` package.

The goal of this project is to build a working understanding of both the applications of foundational machine learning algorithms, while developing a strong understanding of the theory and the math behind each of the algorithms. By implementing the algorithms directly, I was forced to completely understand the math in order to put it into code.

---

## Algorithms

Listed below are the algorithms that are implemented in this repository

1. Perceptron
2. Gradient Descent
3. Linear Regression
4. Logistic Regression
5. Multilayer Perceptron (Neural Network)
6. K Nearest Neighbors
7. Decision Tree
8. Random Forest
9. K Means Clustering

The first eight of the above algorithms are **Supervised Learning Algorithms** which means the data that they are trained on is already labeled allowing more effective training as the algorithm is able to determine how "correct" it is on its predictions. The final algorithm is an **Unsupervised Learning Algorithm** in which case the data is unlabeled and the algorithm is solely able to distinguish patterns within the features of the data.

---

## Training and Testing

One of the most challenging aspects of training machine learning algorithms is the quantity of data needed for a model to have high accuracy rates. Thankfully, the Scikit-learn package provides numerous different datasets that have been pre-labeled by humans available to train the algorithms on. I will note that labeled data is often needed both to train and test different algorithms, so generally we will load only 75% of the data to train the algorithm and use the remaining 25% to test the accuracy of the algorithm. The datasets used in this repository are:

---

## Classification vs Regression

Many times throughout this repository you may see me refer to an algorithm as either a *Classification Algorithm* or a *Regression Algorithm*. Classification algorithms sort data into a discrete set of classes. For example, I may want to sort a group of fruits into either apples or oranges, which would require a classification algorithm based on some features of the fruit (maybe in this case color or sourness would do the trick). Alternatively, a Regression Algorithm takes input data and assigns a value. An example of this may be an attempt to estimate the price of a house given a certain number of criteria. A regression model would be helpful in this case as it would spit out a number rather than a class. I will refer to these types of algorithms in each notebook.

---

## Programming Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, JupyterNotebook, SciKit
