import numpy as np
from sklearn.base import BaseEstimator

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import load_iris


class EstymatorAL(BaseEstimator):
    def __init__(self, base_estimator, budget, strat):
        self.base_estimator = base_estimator
        self.budget = budget
        self.strat = strat
        self.estimator = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        temp = int(0.3 * n_samples)
        a = np.random.choice(np.arange(n_samples), size=temp)
        X_temp = X[a]
        y_temp = y[a]
        p = np.zeros(n_samples)
        for i in range(len(X)):
            p[i] = i
        b = np.delete(p, a)
        X_rest = X[b.astype(int)]
        y_rest = y[b.astype(int)]
        self.estimator = self.base_estimator.fit(X_temp, y_temp)
        probas = self.base_estimator.predict_proba(X_rest)
        us = self.us_scores(probas, strat=self.strat)
        for i in range(self.budget):
            un_idx = np.argmax(us)
            self.estimator = self.base_estimator.partial_fit([X_rest[un_idx]], [y_rest[un_idx]])
            us[un_idx] = float('-inf')
        return self

    def us_scores(self, pr, strat):
        if strat == 'lc':
            return 1 - np.max(pr, axis=1)
        elif strat == 'ms':
            pro = -(np.partition(-pr, 1, axis=1)[:, :2])
            return 1 - np.abs(pro[:, 0] - pro[:, 1])
        else:
            return 'Wybrano złą strategię'

    def predict(self, X):
        y_pred = self.estimator.predict(X)
        return y_pred


iris = load_iris()
X, y = iris.data, iris.target
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
kfold = rskf.split(X, y)
scores = np.zeros(10)
for k, (train, test) in enumerate(kfold):
    clf = EstymatorAL(GaussianNB(), 15, 'lc')
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores[k] = balanced_accuracy_score(y[test], y_pred)

sc = np.mean(scores)
print('balanced accuracy score: %.3f' % sc)



