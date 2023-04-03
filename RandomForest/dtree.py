import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # in random forest, we can use a weighted average of the predictions from the trees as the final prediction
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)

    def leaf(self, x_test):
        """Given a single test record, x_test, return the leaf node 
        reached by running it down the tree starting at this node.  
        This is just like prediction, except we return the decision 
        tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] < self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        """Note you can use scipy.stats for the mode function"""
        self.n = len(y)
        self.y = y
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        # return rrr
        self.prediction = stats.mode(self.y)[0][0]
        return self.prediction

    def leaf(self, x_test):
        return self


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    return 1 - np.sum((np.unique(x, return_counts=True)[1] / len(x)) ** 2)


def find_best_split(X, y, loss, min_samples_leaf, max_features):
    """
    Find the best split for X,y using loss function
    Return the column and split value
    """
    # case when this node cannot be further split
    if len(X) <= min_samples_leaf or len(np.unique(X)) == 1:
        return -1, -1
    # case when this node is a leaf
    best_col = -1
    best_split = -1
    best_loss = loss(y)
    k = 11
    for col in np.random.choice(X.shape[1], size=int(max_features * X.shape[1]), replace=False):
        candidate = np.random.choice(X[:, col], size=k)
        for split in candidate:
            left = y[X[:, col] < split]
            right = y[X[:, col] >= split]
            if len(left) <= min_samples_leaf or len(right) <= min_samples_leaf:
                continue
            loss_val = loss(left) * len(left) / len(y) + \
                loss(right) * len(right) / len(y)
            if loss_val == 0:
                return col, split
            if loss_val < best_loss:
                best_col = col
                best_split = split
                best_loss = loss_val
    return best_col, best_split


class DecisionTree621:
    def __init__(self, max_features=11, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        # loss function; either np.var for regression or gini for classification
        self.loss = loss
        self.max_features = max_features

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(np.unique(y)) == 1 or len(y) <= self.min_samples_leaf:
            return LeafNode(y, self.create_leaf(y))
        col, split = find_best_split(
            X, y, self.loss, self.min_samples_leaf, self.max_features)
        if col == -1:
            return LeafNode(y, self.create_leaf(y))
        left = X[:, col] < split
        right = X[:, col] >= split
        lchild = self.fit_(X[left], y[left])
        rchild = self.fit_(X[right], y[right])
        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.array([self.root.predict(x_test) for x_test in X_test])

    def get_leaf_node(self, X_test):
        return self.root.leaf(X_test)


class RegressionTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1):
        super().__init__(max_features, min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, max_features, min_samples_leaf=1):
        super().__init__(max_features, min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, stats.mode(y)[0][0])
