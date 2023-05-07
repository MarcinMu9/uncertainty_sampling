# Marcin Muszkieta 259719
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.datasets import make_classification, load_iris
from sklearn.naive_bayes import GaussianNB


class Estymator(BaseEstimator):
    def __init__(self, base_clf):
        self.base_clf = base_clf

    def fit(self, X, y):
        self.base_clf.fit(X, y)
        self.is_fit_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fit_')
        y_pred = self.base_clf.predict(X)

        return y_pred


# Generowanie syntetycznego zbioru danych
# iris = load_iris()
sam = 200  # liczba próbek
B = 10  # budżet (możliwa liczba próbek do sklasyfikowania)
X, y = make_classification(n_samples=sam, n_features=10, n_informative=5, random_state=1)
# X = iris.data
# y = iris.target

# Inicjacja klasyfikatora i estymatora

# clf = DecisionTreeClassifier()
clf = KNeighborsClassifier()
# clf = GaussianNB()
model = Estymator(clf)


# obliczenie niepewności dla wybranej strategii US


def us_scores(pr):
    # wybór strategii 'lc' lub 'ms' (least confidence lub margin sampling)
    strat = 'ms'

    if strat == 'lc':
        return 1 - np.max(pr, axis=1)
    elif strat == 'ms':
        pro = -(np.partition(-pr, 1, axis=1)[:, :2])
        return 1 - np.abs(pro[:, 0] - pro[:, 1])
    else:
        return 'Wybrano złą strategię'


# algorytm Uncertainty Sampling


def uncertainty_sampling(X_pool, y_pool, model, clf, budget):
    X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=1)

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
    ac = np.zeros((budget, 1), dtype=float)
    bac = np.zeros((budget, 1), dtype=float)
    us_ = np.zeros((budget, 1), dtype=float)
    cv_ = np.zeros((budget, 2), dtype=float)
    cv_m = np.zeros((budget, 1), dtype=float)
    cv_s = np.zeros((budget, 1), dtype=float)

    for i in range(budget):
        model.fit(X_train, y_train)

        # obliczenie niepewności
        probas = clf.predict_proba(X_test)
        us = us_scores(probas)

        idx = np.argmax(us)
        x = X_test[idx]
        y = y_test[idx]

        # dodanie próbki z największą niepewnością do zbioru treningowego
        X_train = np.vstack((X_train, x.reshape(1, -1)))
        y_train = np.hstack((y_train, y))

        # usunięcie tej próbki ze zbioru testowego
        X_test = np.delete(X_test, idx, axis=0)
        y_test = np.delete(y_test, idx)

        # zapisanie wyników dla każdej iteracji
        pred = model.predict(X_test)
        ac[i] = accuracy_score(y_test, pred)
        # np.savetxt('ac.csv', ac, header='accuracy score')
        # np.save('ac.npy', ac)
        bac[i] = balanced_accuracy_score(y_test, pred)
        # np.savetxt('bac.csv', bac, header='balanced accuracy score')
        # np.save('bac.npy', bac)
        us_[i] = us.mean()
        np.savetxt('us2.csv', us_, header='uncertainty score (ms)')
        np.save('us2.npy', us)
        results = cross_val_score(clf, X_test, y_test, cv=rskf)
        cv_[i] = results.mean(), results.std()
        # np.savetxt('cv.csv', cv_, delimiter=';', header='accuracy: mean (std)')
        # np.save('cv.npy', cv_)
        cv_m[i] = results.mean()
        cv_s[i] = results.std()

    print('accuracy score: %.3f' % np.mean(ac))
    print('balanced accuracy score: %.3f' % np.mean(bac))
    print('uncertainty score: %.3f' % np.mean(us_))
    print('accuracy: %.3f (%.3f)' % (np.mean(cv_m), np.mean(cv_s)))
    print('Budget: ')
    return budget


# wywołanie algorytmu US wraz z wyświetleniem średnich wyników

print(uncertainty_sampling(X, y, model, clf, B))
print()

# wyświetelenie danych zapisanych w plikach .npy
# data = np.load('bac.npy')
# print(data)
