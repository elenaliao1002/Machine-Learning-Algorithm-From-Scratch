import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import os
import pandas as pd


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    # # BEGIN SOLUTION
    df = pd.read_csv(filename, header=None)
    X = np.array(df.iloc[:, :-1])
    Y = np.array(df.iloc[:, -1].apply(lambda x: -1 if x == 0 else x))
    # with open(filename, 'r') as f:
    #     text = f.readlines()
    # lines = [line.strip().split('\n') for line in text]
    # alldata = np.array([])
    # # print(len(lines))
    # for i in lines:
    #     for line in i:
    #         data = np.array(line.split(',')).astype('float')
    #         alldata = np.concatenate([data, alldata])
    # alldata = alldata.reshape(len(lines), len(data))
    # X = alldata[:, :-1]
    # Y = np.where(alldata[:, -1] == 0, -1, alldata[:, -1])
    # END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an num py matrix X, a array y and num_iter return trees and weights 

    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    # initialize weight wi = 1/N
    init_weight = np.ones(N) / N
    # BEGIN SOLUTION
    weight = init_weight
    for i in range(num_iter):
        dt = DecisionTreeClassifier(max_depth=1, random_state=0)
        dt.fit(X, y, sample_weight=weight)
        y_hat = dt.predict(X)
        error = np.sum(weight[y_hat != y]) / np.sum(weight)
        # print(error)
        alphas = (np.log((1-error)/(error+10**-5)))
        # increase weight to misclassified samples
        weight = np.where(
            y_hat != y, weight * np.exp(alphas*0.5),
            weight * np.exp(-alphas*0.5))
        weight = weight / np.sum(weight+10**-5)
        trees_weights.append(weight)
        trees.append(dt)
    # END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ = X.shape
    y = np.zeros(N)
    # BEGIN SOLUTION
    for i in range(len(trees)):
        y_hat = trees[i].predict(X)
        y += y_hat*trees_weights[i]
    # END SOLUTION
    return np.where(y > 0, 1, -1)
