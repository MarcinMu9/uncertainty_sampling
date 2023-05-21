# Marcin Muszkieta
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.datasets import load_iris


class EstymatorAL(BaseEstimator):
    def __init__(self, base_estimator, budget):
        self.base_estimator = base_estimator
        self.budget = budget
        self.estimator = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        bac = np.zeros((self.budget, 1), dtype=float)
        # rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
        # kfold = rskf.split(X, y)
        #
        # for k, (train, test) in enumerate(kfold):
        #     self.estimator = self.base_estimator.fit(X[train, :], y[train])
        #     probas = self.base_estimator.predict_proba(X[train])
        #     us = self.us_scores(probas)
        #     for i in range(self.budget):
        #         un_idx = np.argmax(us)
        #         print('indeks us: ', un_idx)
        #         self.estimator = self.base_estimator.fit([X[un_idx]], [y[un_idx]])
        #         us[un_idx] = float('-inf')
        self.estimator = self.base_estimator.fit(X_train, y_train)
        probas = self.base_estimator.predict_proba(X_train)
        us = self.us_scores(probas)
        for i in range(self.budget):
            un_idx = np.argmax(us)
            print('indeks max us: ', un_idx)
            self.estimator = self.base_estimator.fit([X_train[un_idx]], [y_train[un_idx]])
            bac[i] = balanced_accuracy_score(y_test, self.predict(X_test))
            us[un_idx] = float('-inf')

        bas = np.mean(bac)
        return bas

    def us_scores(self, pr):
        # wybór strategii 'lc' lub 'ms' (least confidence lub margin sampling)
        strat = 'lc'

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
clf = GaussianNB()
model = EstymatorAL(base_estimator=clf, budget=5)

model.fit(X, y)

predictions = model.predict(X)
accu = accuracy_score(y, predictions)
print("acc score:", accu)
print("bal acc score:", model.fit(X, y))
