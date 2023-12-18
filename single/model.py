import scipy
import sklearn
import sklearn.linear_model
import numpy as np

class LogitRegression(sklearn.linear_model.LinearRegression):
    EPS = 1e-5

    def __init__(self):
        super().__init__()

    def _clip01(self,arr):
        return np.asarray(arr).clip(self.EPS,1-self.EPS)

    def fit(self, x, p):
        p = self._clip01(p)
        y = scipy.special.logit(p)
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return scipy.special.expit(y)