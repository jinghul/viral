# Originally written by Shahar Azulay & Ami Tavory
# https://stackoverflow.com/questions/28727709/

# Modifications done by me, Jinghu Lei

import sklearn
import warnings
import numpy as np

class Stacker(object):
    """
    A transformer applying fitting a predictor `pred` to data in a way
        that will allow a higher-up predictor to build a model utilizing both this 
        and other predictors correctly.

    The fit_transform(self, x, y) of this class will create a column matrix, whose 
        each row contains the prediction of `pred` fitted on other rows than this one. 
        This allows a higher-level predictor to correctly fit a model on this, and other
        column matrices obtained from other lower-level predictors.

    The fit(self, x, y) and transform(self, x_) methods, will fit `pred` on all 
        of `x`, and transform the output of `x_` (which is either `x` or not) using the fitted 
        `pred`.

    Arguments:    
        pred: A lower-level predictor to stack.

        cv_fn: Function taking `x`, and returning a cross-validation object. In `fit_transform`
            th train and test indices of the object will be iterated over. For each iteration, `pred` will
            be fitted to the `x` and `y` with rows corresponding to the
            train indices, and the test indices of the output will be obtained
            by predicting on the corresponding indices of `x`.
    """

    def __init__(self, pred, cv_fn=lambda x: sklearn.model_selection.KFold(n_splits=10).split(x)):
        self._pred, self._cv_fn  = pred, cv_fn


    def multi_fit_transform(self, xs, y, r, r2=None):
        """
        Takes an array of feature vectors = modalities, the ground truth, and a range.
        Returns a vector of results of the classifier trained on each of the feature vectors.
        """
        res = np.zeros((len(y), len(xs)))
        for i in range(len(xs)):
            res[r, i] = self.fit_transform(xs[i][r], y[r])[:,0]

        if r2:
            for i in range(len(xs)):
                res[r2, i] = self.fit(xs[i][r], y[r]).transform(xs[i][r2])
        
        return res

    def fit_transform(self, x, y):
        x_trans = self._train_transform(x, y)

        self.fit(x, y)

        return x_trans

    def fit(self, x, y):
        """
        Same signature as any sklearn transformer.
        """
        self._pred.fit(x, y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn transformer.
        """
        return self._test_transform(x)

    def _train_transform(self, x, y):
        x_trans = np.nan * np.ones((x.shape[0], 1))

        all_te = set()
        for tr, te in self._cv_fn(x):
            all_te = all_te | set(te)
            x_trans[te, 0] = self._pred.fit(x[tr, :], y[tr]).predict(x[te, :]) 
        if all_te != set(range(x.shape[0])):
            warnings.warn('Not all indices covered by Stacker', sklearn.exceptions.FitFailedWarning)

        return x_trans

    def _test_transform(self, x):
        return self._pred.predict(x)