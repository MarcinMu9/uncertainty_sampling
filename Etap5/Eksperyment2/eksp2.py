import numpy as np

from Etap4.Estymator import EstymatorAL
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import load_iris

strateg = {'least confident': 'lc', 'margin sampling': 'ms'}
iris = load_iris()
X, y = iris.data, iris.target
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
kfold = rskf.split(X, y)
scores = np.zeros((len(strateg), 10))

for k, (train, test) in enumerate(kfold):
    for str_id, str_name in enumerate(strateg):
        s = strateg[str_name]
        clf = EstymatorAL(GaussianNB(), 10, s)
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[str_id, k] = balanced_accuracy_score(y[test], y_pred)

np.save('e2_bas', scores)

