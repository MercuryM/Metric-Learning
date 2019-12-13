import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from Split import split_data
#
data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape
#
# L2 norm
# X_test = X_test/np.apply_along_axis(np.linalg.norm, 0, X_test)

def get_acc_score(y_valid, y_q, tot_label_occur):
    recall = 0
    true_positives = 0
    k = 0
    max_rank = 30

    rank_A = np.zeros(max_rank)
    AP_arr = np.zeros(11)

    while (recall < 1) or (k < max_rank):

        if (y_valid[k] == y_q):

            true_positives = true_positives + 1
            recall = true_positives/tot_label_occur
            precision = true_positives/(k+1)

            AP_arr[round((recall - 0.05) * 10)] = precision
            for n in range (k, max_rank):
                rank_A[n] = 1

        k = k + 1

    max_precision = 0
    for i in range(10, -1, -1):

        max_precision = max(max_precision, AP_arr[i])
        AP_arr[i] = max_precision

    AP_ = AP_arr.sum()/11

    return AP_, rank_A



from scipy.spatial import distance
from sklearn.metrics import pairwise


def evaluate_metric(X_query, y_query, X_gallery, y_gallery, metric = 'euclidian', parameters = None):

    rank_accuracies = []
    AP = []
    I, K = X_query.shape
    u = X_query.astype(np.float64)
    v = X_gallery.astype(np.float64)
    # u = X_query
    # v = X_gallery
    y_query = y_query.flatten()
    y_gallery = y_gallery.flatten()

    for query, y_q in zip(range(0, K), y_query):
        q_g_dists = []
        y_valid = []
        for gallery, y_g in zip(range(0, K), y_gallery):
            if query == gallery:
                continue
            else:
                if metric == 'euclidian':
                    dist = distance.euclidean(u[:, query], v[:, gallery])
                elif metric == 'sqeuclidean':
                    dist = distance.sqeuclidean(u[:, query], v[:, gallery])
                elif metric == 'seuclidean':
                        dist = distance.seuclidean(u[:, query], v[:, gallery])
                elif metric == 'minkowski':
                    dist = distance.minkowski(u[:, query], v[:, gallery], parameters)
                elif metric == 'chebyshev':
                    dist = distance.chebyshev(u[:, query], v[:, gallery])
                elif metric == 'chi2':
                    dist = -pairwise.additive_chi2_kernel(u[:, query].reshape(1, -1), v[:, gallery].reshape(1, -1))[0][0]
                elif metric == 'braycurtis':
                    dist = distance.braycurtis(u[:, query], v[:, gallery])
                elif metric == 'canberra':
                    dist = distance.canberra(u[:, query], v[:, gallery])
                elif metric == 'cosine':
                    dist = distance.cosine(u[:, query], v[:, gallery])
                elif metric == 'correlation':
                    dist = distance.correlation(u[:, query], v[:, gallery])
                elif metric == 'mahalanobis':
                    dist = distance.mahalanobis(u[:, query], v[:, gallery], parameters)
                else:
                    raise NameError('Specified metric not supported')
                q_g_dists.append(dist)
                y_valid.append(y_g)

        tot_label_occur = y_valid.count(y_q)

        q_g_dists = np.array(q_g_dists)
        y_valid = np.array(y_valid)

        _indexes = np.argsort(q_g_dists)

        # Sorted distances and labels
        q_g_dists, y_valid = q_g_dists[_indexes], y_valid[_indexes]

        AP_, rank_A = get_acc_score(y_valid, y_q, tot_label_occur)

        AP.append(AP_)

        rank_accuracies.append(rank_A)


    rank_accuracies = np.array(rank_accuracies)

    total = rank_accuracies.shape[0]
    rank_accuracies = rank_accuracies.sum(axis = 0)
    rank_accuracies = np.divide(rank_accuracies, total)

    i = 0
    print ('Accuracies by Rank:')
    while i < rank_accuracies.shape[0]:
        print('Rank ', i+1, ' = %.2f%%' % (rank_accuracies[i] * 100), '\t',
              'Rank ', i+2, ' = %.2f%%' % (rank_accuracies[i+1] * 100), '\t',
              'Rank ', i+3, ' = %.2f%%' % (rank_accuracies[i+2] * 100), '\t',
              'Rank ', i+4, ' = %.2f%%' % (rank_accuracies[i+3] * 100), '\t',
              'Rank ', i+5, ' = %.2f%%' % (rank_accuracies[i+4] * 100))
        i = i + 5

    AP = np.array(AP)

    mAP = AP.sum()/AP.shape[0]
    print('mAP = %.2f%%' % (mAP * 100))

    return rank_accuracies, mAP



rank_accuracies_l = []
mAP_l = []
metric_l = []


# histogram

i = 0
histogram1 = []
histogram2 = []

for i in range(0, K):
    arr1 =X_test[:,i].flatten()
    first_edge, last_edge = 0, 255
    n_equal_bins = 255
    bin_edges = np.linspace(start=first_edge, stop=last_edge,
        num = n_equal_bins + 1, endpoint=True)
    H1,_ = np.histogram(arr1, bin_edges)
    histogram1.append(H1)
hist_test = np.array(histogram1)

for j in range(0, N):
    arr2 = X_train[:,j].flatten()
    first_edge, last_edge = 0, 255
    n_equal_bins = 255
    bin_edges = np.linspace(start=first_edge, stop=last_edge,
        num=n_equal_bins + 1, endpoint=True)
    H2,_ = np.histogram(arr2, bin_edges)
    histogram2.append(H2)

hist_train = np.array(histogram2)

X_test = hist_test.T

X_train = hist_train.T

# print(hist)
# plt.figure("test")
print(bin_edges)
# n, bins, patches = plt.hist(arr, bins = bin_edges, normed = 0, alpha=0.75)
# print(bins)
# plt.show()


rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='euclidian',
                                       parameters = None)
rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Baseline')

from sklearn.decomposition import PCA
pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train.T)
X_test_pca = pca.transform(X_test.T)

X_test_pca = X_test_pca.T

print('PCA\n')

# PCA
rank_accuracies, mAP = evaluate_metric(X_test_pca, y_test, X_test_pca, y_test,
                                       metric ='euclidian',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('PCA (M = 200)')


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
print('PCA-LDA\n')


pca2 = PCA(n_components=28)
X_train_pca2 = pca2.fit_transform(X_train.T)
X_test_pca2 = pca2.transform(X_test.T)

X_test_pca2 = X_test_pca2.T

lda = LinearDiscriminantAnalysis(n_components = 28)
X_train_lda = lda.fit_transform(X_train_pca2, y_train.T.ravel())
X_test_lda = lda.transform(X_test_pca2.T)
X_test_lda = X_test_lda.T

# LDA Euclidian
rank_accuracies, mAP = evaluate_metric(X_test_lda, y_test, X_test_lda, y_test,
                                       metric ='euclidian',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('PCA-LDA (M = 25, 25)')
#
plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
for i in range(len(metric_l)):
    plt.plot(np.arange(1, 31), 100*rank_accuracies_l[i], color=color_list[i], linestyle='dashed', label='Metric: '+ metric_l[i])

plt.title('CMC Curves for different features')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy [%]')
plt.legend(loc='best')
plt.savefig('img/CMC_curves_for_features_hist.png', dpi=1000, transparent=True)
# #
#
#
# mAP_array = np.load ('mAP.npy')
# #
# import pandas as pd
# #
# text = pd.DataFrame(mAP_array)
# text.stack().max()
# text.stack().idxmax()
# a = np.max(mAP_array)
# # Square Euclidian
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric = 'sqeuclidean',
#                                        parameters = None)

#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Square Euclidian')
#
#
#
#
# #Manhattan Distance
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric = 'minkowski',
#                                        parameters = 1)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Manhattan')
#
#
#
# # Chebyshev - L_infinity
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='chebyshev',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Chebyshev')
#
#
#
# # Chi-Square
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='chi2',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Chi Square')
#
#
#
#
# # Braycurtis
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='braycurtis',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Bray Curtis')
#
#
#
#
# # Canberra
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='canberra',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Canberra')
#
#
#
# # Cosine
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='cosine',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Cosine')
#
#
# # Correlation
#
# rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
#                                        metric ='correlation',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('Correlation')
#
#
# plt.figure(figsize=(8.0, 6.0))
# color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
# for i in range(len(metric_l)):
#     plt.plot(np.arange(1, 31), 100*rank_accuracies_l[i], color=color_list[i], linestyle='dashed', label='Metric: '+ metric_l[i])
#
#
#
# plt.title('CMC Curves for a range of standard distance metrics')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics.png', dpi=1000, transparent=True)
#