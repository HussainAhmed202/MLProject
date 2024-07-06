import operator

import numpy as np
import pandas as pd

# without packages
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances


# custom KNN transformer
class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neigbours=3):
        self.n_neigbours = n_neigbours
        self.results = np.array([], dtype=int)
        self.classCount = {}

    def fit(self, X_train, y_train):
        self.X_train = X_train.values
        self.y_train = y_train.values
        return self

    def predict(self, X_test):
        if isinstance(X_test, pd.Series) or isinstance(X_test, pd.DataFrame):
            self.X_test = X_test.values.reshape(1, -1)
        else:
            self.X_test = X_test.values
        for row in range(len(self.X_test)):
            test_row = self.X_test[row].reshape(1, -1)
            dist = euclidean_distances(self.X_train, test_row).reshape(-1)
            sort_dist = dist.argsort()
            top_k = sort_dist[: self.n_neigbours]
            for i in top_k:
                voteIlabel = self.y_train[i]
                self.classCount[voteIlabel] = self.classCount.get(voteIlabel, 0) + 1

            sortedClassCount = sorted(
                self.classCount.items(), key=operator.itemgetter(1), reverse=True
            )
            self.results = np.append(self.results, sortedClassCount[0][0])
            self.classCount = {}

        return self.results

    def __str__(self) -> str:
        return f"KNNTransformer(n_neigbours = {self.n_neigbours})"


# custom LR transformer
class LRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.results = np.array([], dtype=int)
        self.classCount = {}

    def fit(self, X, y):

        n_samples, n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate output variable (y) with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(
                X.T, (y_predicted - y)
            )  # derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # derivative w.r.t bias
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values.reshape(1, -1)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def __str__(self) -> str:
        return f"LRTransformer(learning_rate = {self.learning_rate}, n_iters = {self.n_iters})"
