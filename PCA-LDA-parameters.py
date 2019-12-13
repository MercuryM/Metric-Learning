import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from Split import split_data

from mpl_toolkits import mplot3d

#
data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape
# #
# # L2 norm
# X_test = X_test/np.apply_along_axis(np.linalg.norm, 0, X_test)
#
def get_acc_score(y_valid, y_q, tot_label_occur):
    recall = 0
    true_positives = 0
    k = 0
    max_rank = 30

    # rank_A = np.zeros(max_rank)
    AP_arr = np.zeros(11)

    while (recall < 1) or (k < max_rank):

        if (y_valid[k] == y_q):

            true_positives = true_positives + 1
            recall = true_positives/tot_label_occur
            precision = true_positives/(k+1)

            AP_arr[round((recall - 0.05) * 10)] = precision
            # for n in range (k, max_rank):
            #     rank_A[n] = 1

        k = k + 1

    max_precision = 0
    for i in range(10, -1, -1):

        max_precision = max(max_precision, AP_arr[i])
        AP_arr[i] = max_precision

    AP_ = AP_arr.sum()/11

    return AP_


#
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

        AP_ = get_acc_score(y_valid, y_q, tot_label_occur)

        AP.append(AP_)

        # rank_accuracies.append(rank_A)


    # rank_accuracies = np.array(rank_accuracies)
    #
    # total = rank_accuracies.shape[0]
    # rank_accuracies = rank_accuracies.sum(axis = 0)
    # rank_accuracies = np.divide(rank_accuracies, total)

    # i = 0
    # print ('Accuracies by Rank:')
    # while i < rank_accuracies.shape[0]:
    #     print('Rank ', i+1, ' = %.2f%%' % (rank_accuracies[i] * 100), '\t',
    #           'Rank ', i+2, ' = %.2f%%' % (rank_accuracies[i+1] * 100), '\t',
    #           'Rank ', i+3, ' = %.2f%%' % (rank_accuracies[i+2] * 100), '\t',
    #           'Rank ', i+4, ' = %.2f%%' % (rank_accuracies[i+3] * 100), '\t',
    #           'Rank ', i+5, ' = %.2f%%' % (rank_accuracies[i+4] * 100))
    #     i = i + 5

    AP = np.array(AP)

    mAP = AP.sum()/AP.shape[0]
    # print('mAP = %.2f%%' % (mAP * 100))

    return mAP
#
#
#
rank_accuracies_l = []
mAP_l = []
metric_l = []
#
#
# # histogram
#
# i = 0
# histogram=[]
#
# for i in range(0, K):
#     arr=X_test[:,i].flatten()
#     first_edge, last_edge = 0, 256
#     n_equal_bins = 10
#     bin_edges = np.linspace(start=first_edge, stop=last_edge,
#         num=n_equal_bins + 1, endpoint=True)
#     H,_ = np.histogram(arr, bin_edges)
#     histogram.append(H)
# hist = np.array(histogram)
#
#
# # X_test = hist.T
#
#
# # print(hist)
# # plt.figure("test")
# # print(bin_edges)
# # n, bins, patches = plt.hist(arr, bins = bin_edges, normed = 0, alpha=0.75)
# # print(bins)
# # plt.show()
#
# _card = 32

M_pca = 1
M_lda = 1

# M_pca_range = N - _card
# M_lda_range = _card - 1

Ms = np.arange(1, N)
M_pca_range = N - 1
#
from sklearn.decomposition import PCA
#
while M_pca <= M_pca_range:
    print(M_pca)
    pca = PCA(M_pca)
    X_train_pca = pca.fit_transform(X_train.T)
    X_test_pca = pca.transform(X_test.T)
    X_test_pca = X_test_pca.T
    # print('PCA\n')
    # Baseline Euclidia
    mAP = evaluate_metric(X_test_pca, y_test, X_test_pca, y_test,
                                       metric ='euclidian',
                                       parameters = None)
    # rank_accuracies_l.append(rank_accuracies)
    mAP_l.append(mAP)
    # metric_l.append('PCA-Euclidian')
    M_pca = M_pca + 1

# plt.figure(figsize=(8.0, 6.0))
# plt.plot(Ms, mAP_l)
# plt.title(
#         'mAP versus $\mathcal{M}$\n')
# plt.xlabel('$\mathcal{M}$: number of principle components')
# plt.ylabel('Recognition Accuracy [%]')
# plt.savefig('img/mAP_versus_M.png',
#                dpi=1000, transparent=True)
#
plt.figure(figsize=(8.0, 6.0))
plt.plot(Ms, mAP_l)
plt.title(
        'mAP versus $\mathcal{M}$\n')
plt.xlabel('$\mathcal{M}$: number of principle components')
plt.ylabel('mAP')
plt.savefig('img/mAP_versus_M.png',
                dpi=1000, transparent=True)
#
# mAP_array = np.zeros((M_pca_range , M_lda_range))
# mAP_max = 0
#
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# while M_pca <= M_pca_range:
#     pca = PCA(M_pca)
#     X_train_pca = pca.fit_transform(X_train.T)
#     X_test_pca = pca.transform(X_test.T)
#     X_test_pca = X_test_pca.T
#     M_lda = 1
#     while M_lda <= M_lda_range and M_lda <= M_pca:
#         lda = LinearDiscriminantAnalysis(n_components=M_lda)
#         X_train_lda = lda.fit_transform(X_train_pca, y_train.T.ravel())
#         X_test_lda = lda.transform(X_test_pca.T)
#         X_test_lda = X_test_lda.T
#         mAP = evaluate_metric(X_test_lda, y_test, X_test_lda, y_test,
#                                        metric ='euclidian',
#                                        parameters = None)
#         mAP_array[M_pca - 1, M_lda - 1] = mAP
#         print('M_pca = ', M_pca, ', M_lda = ', M_lda, ' --->  mAP = %.2f%%' % (mAP * 100))
#         # if (mAP > mAP_max):
#         #     M__pca_ideal = M_pca
#         #     M__lda_ideal = M_lda
#         #     mAP_max = mAP
#         # rank_accuracies_l.append(rank_accuracies)
#         mAP_l.append(mAP)
#         # metric_l.append('PCA-Euclidian')
#         M_lda = M_lda + 1
#     M_pca = M_pca + 1
# # print("mAP is maximum for M__pca = ", M__pca_ideal, ", M_lda = ", M__lda_ideal,
# #       " with accuracy of %.2f%%" % (mAP_max * 100), ".")
#
# np.save('mAP.npy',mAP_array)
# M_lda_range = 32 -  1
# M_pca_range = 320 - 32
# x = np.linspace(1, M_lda_range, M_lda_range)
# y = np.linspace(1, M_pca_range, M_pca_range)
#
#
# mAP_array = np.load('mAP.npy')
# #
# for i in range(0,M_pca_range):
#     j = 0
#     for j in range(0, M_lda_range):
#         if (j > i):
#             mAP_array[i,j] = mAP_array[i,i]
#         else:
#             mAP_array[i,j] = mAP_array[i,j]
#
# X, Y = np.meshgrid(x, y)
#
# print(mAP_array.shape)
# print(X.shape)
# print(Y.shape)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, mAP_array, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('mAP varying M_pca & M_lda');
# ax.set_xlabel('M_lda')
# ax.set_ylabel('M_pca')
# ax.set_zlabel('mAP');
#
# ax.view_init(30, 220)
# plt.savefig('img/mAP_versus_M_pca&M_lda.png', dpi=1000, transparent=True)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# print('PCA-LDA\n')
#
# lda = LinearDiscriminantAnalysis(n_components = 30)
# X_train_lda = lda.fit_transform(X_train_pca, y_train.T.ravel())
# X_test_lda = lda.transform(X_test_pca.T)
# X_test_lda = X_test_lda.T
#
# # Baseline Euclidian
# rank_accuracies, mAP = evaluate_metric(X_test_lda, y_test, X_test_lda, y_test,
#                                        metric ='euclidian',
#                                        parameters = None)
#
# rank_accuracies_l.append(rank_accuracies)
# mAP_l.append(mAP)
# metric_l.append('LDA-Euclidian')

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

