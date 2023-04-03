import numpy as np
from sklearn.utils import resample

from dtree import *


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators  # number of trees in the forest
        self.oob_score = oob_score  # whether to compute OOB score
        self.oob_score_ = np.nan  # OOB score estimate

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn. 
        """
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))

        self.nunique = len(np.unique(y))
        self.trees = []
        self.oob_indexes = []
        for i in range(self.n_estimators):
            # we cannot get the oob_index from the resample function
            # and we cannot use the booststrap function in sklearn
            # so we have to use the resample function in sklearn.utils
            # and then use the np.setdiff1d to track the oob_index
            X_resample, y_resample = resample(X, y)
            oob_index = set(range(len(X))) - set(np.where(X_resample == X)[0])
            # set diff1d is the difference between two arrays and return the unique values in the first array
            # the first array is the index of X and the second array is the index of X_resample
            # so we can get the oob_index
            self.oob_indexes.append(oob_index)
            tree = RegressionTree621(self.max_features, self.min_samples_leaf)
            tree.fit(X_resample, y_resample)
            self.trees.append(tree)

        if self.oob_score:
            self.compute_oob_score(X, y)

    def compute_oob_score(self, X, y):
        """
        Given an (X, y) training set, compute the OOB validation score estimate
        and store as self.oob_score_.  This is the average R^2 score across all
        trees in the forest, computed on the OOB records for each tree. 
        """
        # Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required
        # this is the error I got when I tried to use the r2_score function
        # so I have to use the mean_squared_error function
        # and then use the r2_score function to get the r2 score
        # but I still don't know why I got the error
        # I think it is because the oob_index is empty
        # so I have to use the if statement to check if the oob_index is empty
        # if it is empty, then I will not compute the oob_score
        # and I will set the oob_score to be nan
        y_test = []
        y_pred = []
        for i, obs in enumerate(zip(X, y)):
            x_obs, y_obs = obs
            leaves = [tree.root.leaf(x_obs)
                      for tree in self.trees if i in self.oob_indexes]
            if leaves:
                preds = np.sum([leaf.prediction * leaf.n for leaf in leaves]
                               ) / np.sum([leaf.n for leaf in leaves])
                y_pred.append(preds)
                y_test.append(y_obs)
        self.oob_score_ = r2_score(y_test, y_pred)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        # andomForest621.fit() calls self.compute_oob_score()
        # and that calls the implementation either in regressor or
        # classifier,depending on which object I created
        self.trees = []
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test. 
        """
        # fist we need to get the prediction for each tree
        # take all the y from each leaf and then get the weighted average
        # weight each trees prediction by the number of observations in the leaf making that prediction
        # return a 1D vector with the predictions for each input record of X_test
        y_pred = []
        for x in X_test:
            pred_leaves = np.array([tree.root.leaf(x) for tree in self.trees])
            preds = [np.mean(leaf.y) for leaf in pred_leaves]
            weights = [leaf.n for leaf in pred_leaves]
            y_pred.append(np.average(preds, weights=weights))
        return np.array(y_pred)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        # compute R^2 on that and y_test
        return r2_score(y_test, y_pred)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.max_features = max_features
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def predict(self, X_test) -> np.ndarray:
        """for the random forest classifier, we need to get the prediction for each tree
        and then get the weighted average prediction from all trees in this forest
        weight each trees prediction by the number of observations in the leaf making that prediction
        return a 1D vector with the predictions for each input record of X_test
        """
        pred = []
        for i in range(self.n_estimators):
            pred.append(self.trees[i].predict(X_test))
        pred = np.array(pred)
        pred = np.array([np.bincount(pred[:, i], minlength=self.nunique)
                        for i in range(pred.shape[1])])
        pred = np.array([np.argmax(pred[i]) for i in range(pred.shape[0])])
        return pred

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        # compute accuracy between that and y_test
        accuracy = np.mean(y_test == y_pred)
        return accuracy
