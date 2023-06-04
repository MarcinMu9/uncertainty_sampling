import numpy as np
from sklearn.neural_network import MLPClassifier

from Etap4.Estymator import EstymatorAL
from sklearn import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification

clfs = {'GNB': GaussianNB(), 'MLP': MLPClassifier()}
X, y = make_classification(n_samples=600, n_features=20, n_informative=15, random_state=1)
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
kfold = rskf.split(X, y)
scores = np.zeros((len(clfs), 10))
prec = np.zeros((len(clfs), 10))
# rec = np.zeros((len(clfs), 10))

for k, (train, test) in enumerate(kfold):
    for clf_id, clf_name in enumerate(clfs):
        c = clone(clfs[clf_name])
        clf = EstymatorAL(c, 5, 'lc')
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[clf_id, k] = balanced_accuracy_score(y[test], y_pred)
        prec[clf_id, k] = precision_score(y[train], y_pred)
        # rec[clf_id, k] = recall_score(y[train], y_pred)

np.save('e1_bas', scores)
np.save('e1_prec', prec)
# np.save('e1_recall', rec)
