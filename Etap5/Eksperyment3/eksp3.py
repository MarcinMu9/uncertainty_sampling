import numpy as np

from Etap4.Estymator import EstymatorAL
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification

bud = {'1': 1, '5': 5, '10': 10, '15': 15, '20': 20, '30': 30, '40': 40, 50: 50, '75': 75,
       '100': 100, '125': 125, '150': 150, '200': 200}
X, y = make_classification(n_samples=800, n_features=40, n_informative=20, random_state=1)
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
kfold = rskf.split(X, y)
scores = np.zeros((len(bud), 10))
prec = np.zeros((len(bud), 10))
rec = np.zeros((len(bud), 10))

for k, (train, test) in enumerate(kfold):
    for bud_id, bud_name in enumerate(bud):
        b = bud[bud_name]
        clf = EstymatorAL(GaussianNB(), b, 'lc')
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[bud_id, k] = balanced_accuracy_score(y[test], y_pred)
        prec[bud_id, k] = precision_score(y[train], y_pred)
        rec[bud_id, k] = recall_score(y[train], y_pred)

np.save('e3_bas', scores)
np.save('e3_prec', prec)
# np.save('e3_recall', rec)
