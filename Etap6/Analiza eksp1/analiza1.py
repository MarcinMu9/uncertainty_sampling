import numpy as np
from Etap5.Eksperyment1.eksp1 import clfs
from scipy import stats


scores = np.load('e1_bas.npy')
res1 = scores
# print(scores)

sc1 = np.mean(scores[0])
sc1s = np.std(scores[0])
sc2 = np.mean(scores[1])
sc2s = np.std(scores[1])

print('balanced accuracy score GNB: %.3f (%.3f)' % (sc1, sc1s))
print('balanced accuracy score MLP: %.3f (%.3f)' % (sc2, sc2s))

alfa = 0.05
t_statistic = np.zeros(((len(clfs)), (len(clfs))))
p_value = np.zeros(((len(clfs)), (len(clfs))))
better = np.zeros(((len(clfs)), (len(clfs)))).astype(bool)

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = stats.ttest_ind(res1[i], res1[j])
        better[i, j] = np.mean(res1[i]) > np.mean(res1[j])

significant = p_value < alfa
sign_better = significant * better

print(significant)
print(sign_better)

prec = np.load('e1_prec.npy')
# print(prec)
pr1 = np.mean(prec[0])
pr1s = np.std(prec[0])
pr2 = np.mean(prec[1])
pr2s = np.std(prec[1])
print('precision score GNB: %.3f (%.3f)' % (pr1, pr1s))
print('precision score MLP: %.3f (%.3f)' % (pr2, pr2s))

# rec = np.load('e1_recall.npy')
# # print(rec)
# re1 = np.mean(rec[0])
# re1s = np.std(rec[0])
# re2 = np.mean(rec[1])
# re2s = np.std(rec[1])
# print('recall score GNB: %.3f (%.3f)' % (re1, re1s))
# print('recall score MLP: %.3f (%.3f)' % (re2, re2s))
