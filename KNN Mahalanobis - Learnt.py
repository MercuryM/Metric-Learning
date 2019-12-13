import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from Split import split_data


data = split_data()
X_train, y_train = data['train']
D, N = X_train.shape
X_test, y_test = data['test']
I, K = X_test.shape

# L2 norm
# X_train = X_train/np.apply_along_axis(np.linalg.norm, 0, X_train)
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

def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u

def mahalanobis(u, v, VI):
    """
    Compute the Mahalanobis distance between two 1-D arrays.

    The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       \\sqrt{ (u-v) V^{-1} (u-v)^T }

    where ``V`` is the covariance matrix.  Note that the argument `VI`
    is the inverse of ``V``.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    VI : ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `u` and `v`.
    """
    u = np.array(u)
    v = np.array(v)
    # VI = np.atleast_2d(VI)
    delta = u - v
#    print(delta)
    m = np.dot(np.dot(delta, VI), delta.T)
    return m



from scipy.spatial import distance
from sklearn.metrics import pairwise


def evaluate_metric(X_query, y_query, X_gallery, y_gallery, metric, parameters):

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
                elif metric == 'mahalanobis':
                    dist = mahalanobis(u[:, query], v[:, gallery], parameters)
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
    
        #if q  > 5:
        #    break
        #q = q+1

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
        i = i+5

    AP = np.array(AP)

    mAP = AP.sum()/AP.shape[0]
    print('mAP = %.2f%%' % (mAP * 100))
    
    return rank_accuracies, mAP

rank_accuracy_base = np.array(
        [63.50, 76.50, 83.50, 86.00, 88.50, 90.00, 93.00, 93.50, 93.50, 94.50, 96.00, 96.00, 96.00, 96.50, 96.50, 96.50, 96.50,
         96.50, 97.00, 97.00, 97.50, 98.00, 98.00, 98.00, 98.00, 98.00, 98.00, 98.50, 99.00, 99.00])


rank_accuracies_l_2 = []
mAP_l_2 = []
metric_l_2 = []


# Mahalanobis - inverse covariance

mean_face = X_train.mean(axis = 1).reshape(-1, 1)
V = np.cov(X_train - mean_face)
print(V.shape)
VI = np.linalg.pinv(V)
# print(VI)
print (VI.shape)
rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='mahalanobis',
                                       parameters = VI)

rank_accuracies_l_2.append(rank_accuracies)
mAP_l_2.append(mAP)
metric_l_2.append('Mahalanobis - Covariance')

# #
from metric_learn import MMC_Supervised
from sklearn.decomposition import PCA
# # #
# # #Mahalanobis - learnt - reduced set
# #
# M = [16, 32, 64, 128, 256]
# for m in M:
#     pca = PCA(n_components=m)
#     X_train_pca = pca.fit_transform(X_train.T)
#     X_test_pca = pca.transform(X_test.T)
#     X_test_pca = X_test_pca.T
#     mean_face_pca = X_train_pca.mean(axis = 1).reshape(-1, 1)
#     V = np.cov(X_train_pca.T - mean_face_pca.T)
#     print(V.shape)
#     VI_pca = np.linalg.pinv(V)
#     rank_accuracies, mAP = evaluate_metric(X_test_pca, y_test, X_test_pca, y_test,
#                                            metric='mahalanobis',
#                                            parameters=VI_pca)
#
#     rank_accuracies_l_2.append(rank_accuracies)
#     mAP_l_2.append(mAP)
# metric_l_2.append('Mahalanobis - 16')
# metric_l_2.append('Mahalanobis - 32')
# metric_l_2.append('Mahalanobis - 64')
# metric_l_2.append('Mahalanobis - 128')
# metric_l_2.append('Mahalanobis - 256')


pca = PCA(n_components=150)
X_train_pca = pca.fit_transform(X_train.T)
X_test_pca = pca.transform(X_test.T)

# mmc = MMC_Supervised(max_iter=50)
# mmc.fit(X_train_pca, y_train.T)
#
# M = mmc.metric()
#
# print ('Metric learnt')
#
#
# rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test,
#                                        X_test_pca.T, y_test,
#                                        metric ='mahalanobis',
#                                        parameters = M)
# #
# rank_accuracies_l_2.append(rank_accuracies)
# mAP_l_2.append(mAP)
# metric_l_2.append('Learnt Mahalanobis (Red. Set)')
# #
#

#
#
import metric_learn
#
lmnn = metric_learn.LMNN(k=3, learn_rate=1e-6, max_iter=50)
lmnn.fit(X_train_pca, y_train.T)
M = lmnn.metric()

print ('Metric learnt-LMNN')

rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test, X_test_pca.T, y_test,
                                       metric ='mahalanobis',
                                       parameters = M)

rank_accuracies_l_2.append(rank_accuracies)
mAP_l_2.append(mAP)
metric_l_2.append('Learnt LMNN')

#
import metric_learn

NCA = metric_learn.NCA( max_iter=10)
NCA.fit(X_train_pca, y_train.T)

N = NCA.metric()

print ('Metric learnt-NCA')

rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test, X_test_pca.T, y_test,
                                       metric ='mahalanobis',
                                       parameters = N)

rank_accuracies_l_2.append(rank_accuracies)
mAP_l_2.append(mAP)
metric_l_2.append('Learnt NCA')



import metric_learn
mlkr = metric_learn.MLKR()
mlkr.fit(X_train_pca, y_train.T)
ML = mlkr.metric()
print ('Metric learnt-mlkr')

rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test, X_test_pca.T, y_test,
                                       metric ='mahalanobis',
                                       parameters = ML)

rank_accuracies_l_2.append(rank_accuracies)
mAP_l_2.append(mAP)
metric_l_2.append('Learnt MLKR')



lsml = metric_learn.LSML_Supervised(num_constraints=200)
lsml.fit(X_train_pca, y_train.T)
LS = lsml.metric()
print ('Metric learnt-sdml')

rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test, X_test_pca.T, y_test,
                                       metric ='mahalanobis',
                                       parameters = LS)

rank_accuracies_l_2.append(rank_accuracies)
mAP_l_2.append(mAP)
metric_l_2.append('Learnt SDML')

import metric_learn
#
#
# RCA = metric_learn.RCA_Supervised(n_components=150, num_chunks=30, chunk_size=2)
# RCA.fit(X_train_pca, y_train.T)
#
# R = RCA.metric()
#
# print ('Metric learnt-RCA')
#
# rank_accuracies, mAP = evaluate_metric(X_test_pca.T, y_test, X_test_pca.T, y_test,
#                                        metric ='mahalanobis',
#                                        parameters = R)
#
# rank_accuracies_l_2.append(rank_accuracies)
# # mAP_l_2.append(mAP)
# metric_l_2.append('Learnt RCA')

#
#
# # In[29]:
#

# plt.figure(figsize=(8.0, 6.0))
# color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
# for i in range(len(metric_l_2)):
#     plt.plot(np.arange(1, 31), 100*rank_accuracies_l_2[i], color=color_list[i], linestyle='dashed', label=metric_l_2[i])
# plt.plot(np.arange(1, 31), rank_accuracy_base, color='darkorange', linestyle=':', label='kNN Baseline')
# plt.title('CMC Curves for Mahalanobis method versus dimensions')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy / %')
# plt.legend(loc='best')
# plt.savefig('img/CMC_Curves_for_Mahalanobis_method_versus_dimensions.png', dpi = 1000, transparent = True)



plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
for i in range(len(metric_l_2)):
    plt.plot(np.arange(1, 31), 100*rank_accuracies_l_2[i], color=color_list[i], linestyle='dashed', label=metric_l_2[i])
plt.plot(np.arange(1, 31), rank_accuracy_base, color='darkorange', linestyle=':', label='kNN Baseline')
plt.title('CMC Curves for various Mahalanobis methods')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy / %')
plt.legend(loc='best')
plt.savefig('img/CMC_Curves_for_various_Mahalanobis_methods_ALL.png', dpi = 1000, transparent = True)

# plt.figure(figsize=(8.0, 6.0))
# color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
# for i in range(len(metric_l_2)):
#     plt.plot(np.arange(1, 31), 100*rank_accuracies_l_2[i], color=color_list[i], linestyle='dashed', label=metric_l_2[i])
# plt.plot(np.arange(1, 31), rank_accuracy_base, color='darkorange', linestyle=':', label='kNN Baseline')
# plt.title('CMC Curves for various Mahalanobis methods')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy / %')
# plt.legend(loc='best')
# plt.savefig('img/CMC_Curves_for_various_Mahalanobis_methods.png', dpi = 1000, transparent = True)
