import numpy as np

#import seaborn as sns
#import matplotlib
#import matplotlib.pyplot as plt
from Split import split_data
#

#
# L2 norm
# X_test = X_test/np.apply_along_axis(np.linalg.norm, 0, X_test)

#def get_acc_score(y_valid, y_q, tot_label_occur):
#    recall = 0
#    true_positives = 0
#    k = 0
#    j = 0
#    max_rank = 30
#
#    rank_A = np.zeros(max_rank)
#    AP_arr = np.zeros(9)
#
#    while (recall < 1) or (k < max_rank):
#
#        if (y_valid[k] == y_q):
#
#            true_positives = true_positives + 1
#            recall = true_positives/tot_label_occur
#            precision = true_positives/(k+1)
#
#            AP_arr[j] = precision
#            j = j + 1
#            for n in range (k, max_rank):
#                rank_A[n] = 1
#
#        k = k + 1
#
#    max_precision = 0
#    for i in range(0, j):
#
#        max_precision = max(max_precision, AP_arr[i])
#        AP_arr[i] = max_precision
##    print(j)
#    AP_ = AP_arr.sum()/j
#
#    return AP_, rank_A


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



