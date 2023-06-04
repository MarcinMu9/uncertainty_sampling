import numpy as np
from Etap5.Eksperyment3.eksp3 import bud
from scipy import stats
import matplotlib.pyplot as plt


scores = np.load('e3_bas.npy')
res3 = scores
# print(scores)

sc_m = np.zeros(len(bud))
sc_s = np.zeros(len(bud))
b = np.zeros(len(bud))

for i, bud_name in enumerate(bud):
    sc_m[i] = np.mean(scores[i])
    sc_s[i] = np.std(scores[i])
    b[i] = bud[bud_name]
    print('balanced accuracy score (Budget: %3s): %.3f (%.3f)' % (bud_name, sc_m[i], sc_s[i]))
plt.plot(b, sc_m)
plt.xlabel("Budżet")
plt.ylabel("Balanced Accuracy Score")
plt.title("Wartość BAS w zależności od zadanego budżetu")
plt.savefig('e3_bas.png')
# sc1 = np.mean(scores[0])
# sc1s = np.std(scores[0])
# sc2 = np.mean(scores[1])
# sc2s = np.std(scores[1])
#
# print('balanced accuracy score GNB: %.3f (%.3f)' % (sc1, sc1s))
# print('balanced accuracy score MLP: %.3f (%.3f)' % (sc2, sc2s))

alfa = 0.05
t_statistic = np.zeros(((len(bud)), (len(bud))))
p_value = np.zeros(((len(bud)), (len(bud))))
better = np.zeros(((len(bud)), (len(bud)))).astype(bool)

for i in range(len(bud)):
    for j in range(len(bud)):
        t_statistic[i, j], p_value[i, j] = stats.ttest_ind(res3[i], res3[j])
        better[i, j] = np.mean(res3[i]) > np.mean(res3[j])

significant = p_value < alfa
sign_better = significant * better

# print(significant)
# print(sign_better)

prec = np.load('e3_prec.npy')
pr_m = np.zeros(len(bud))
pr_s = np.zeros(len(bud))

for i, bud_name in enumerate(bud):
    pr_m[i] = np.mean(prec[i])
    pr_s[i] = np.std(prec[i])
    print('precision score (Budget: %3s): %.3f (%.3f)' % (bud_name, pr_m[i], pr_s[i]))
plt.plot(b, pr_m)
plt.xlabel("Budżet")
plt.ylabel("Precision Score")
plt.title("Wartość Precision w zależności od zadanego budżetu")
plt.savefig('e3_prec.png')

# rec = np.load('e3_recall.npy')
# re_m = np.zeros(len(bud))
# re_s = np.zeros(len(bud))
#
# for i, bud_name in enumerate(bud):
#     re_m[i] = np.mean(rec[i])
#     re_s[i] = np.std(rec[i])
#     print('recall score (Budget: %3s): %.3f (%.3f)' % (bud_name, re_m[i], re_s[i]))
# plt.plot(b, re_m)
# plt.xlabel("Budżet")
# plt.ylabel("Recall Score")
# plt.title("Wartość Recall w zależności od zadanego budżetu")
# plt.savefig('e3_recall.png')
