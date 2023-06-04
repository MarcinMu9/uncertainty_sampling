import numpy as np
from Etap5.Eksperyment2.eksp2 import strateg
from scipy import stats


scores = np.load('e2_bas.npy')
res2 = scores
print(scores)

sc1 = np.mean(scores[0])
sc1s = np.std(scores[0])
sc2 = np.mean(scores[1])
sc2s = np.std(scores[1])

print('balanced accuracy score Least Confidence: %.3f (%.3f)' % (sc1, sc1s))
print('balanced accuracy score Margin Sampling: %.3f (%.3f)' % (sc2, sc2s))

alfa = 0.05
t_statistic = np.zeros(((len(strateg)), (len(strateg))))
p_value = np.zeros(((len(strateg)), (len(strateg))))
better = np.zeros(((len(strateg)), (len(strateg)))).astype(bool)

for i in range(len(strateg)):
    for j in range(len(strateg)):
        t_statistic[i, j], p_value[i, j] = stats.ttest_ind(res2[i], res2[j])
        better[i, j] = np.mean(res2[i]) > np.mean(res2[j])

significant = p_value < alfa
sign_better = significant * better

print(significant)
print(sign_better)
