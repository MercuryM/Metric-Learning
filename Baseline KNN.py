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

import cv2
# histogram
from skimage.feature import hog
from skimage import io
for i in range(0,200):
    b = np.reshape(X_test[:,i],(46,56))
    normalised_blocks, x = hog(b, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        block_norm='L2-Hys', visualize=True, transform_sqrt=True,
        feature_vector = True, multichannel=None)
    plt.show(x)
    X_test[:,i] = x.ravel()

# i = 0
# histogram=[]
#
# for i in range(0, K):
#     arr = X_test[:,i].flatten()
#     first_edge, last_edge = 0, 255
#     n_equal_bins = 255
#     bin_edges = np.linspace(start=first_edge, stop=last_edge,
#         num=n_equal_bins + 1, endpoint=True)
#     H,_ = np.histogram(arr, bin_edges)
#     histogram.append(H)
# hist = np.array(histogram)

# X_test = hist.T

#
# print(hist)
# plt.figure("test")
# print(bin_edges)
# n, bins, patches = plt.hist(X_test[:,0].flatten(), bins = bin_edges, normed = 0, alpha = 0.5, label = 'Image 2')
# n, bins, patches = plt.hist(X_test[:,1].flatten(), bins = bin_edges, normed = 0, alpha = 0.5, label = 'Image 2')
# print(bins)
# plt.legend()
# plt.title('Histogram of Image 1 and Image 2')
# plt.show()
# plt.savefig('img/Histogram_of_Image_1_and_Image_2(10).png', dpi=1000, transparent=True)

# x1 = np.arange(10)
# bar_width = 0.35
# print(hist)
# plt.figure("test")
# print(bin_edges)
#
# r1 = plt.bar(x1, X_test[:,0].flatten(), alpha = 0.75, label = 'Image 1')
# r2 = plt.bar(x1 + bar_width,X_test[:,1].flatten(), alpha = 0.75, label = 'Image 2')
# plt.xticks(x1 + bar_width/2, bin_edges)
# # print(bins)
# plt.legend()
# plt.show()





# Baseline Euclidian
rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='euclidian',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Euclidian')



# Square Euclidian
rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric = 'sqeuclidean',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Square Euclidian')




#Manhattan Distance

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric = 'minkowski',
                                       parameters = 1)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Manhattan')



# Chebyshev - L_infinity

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='chebyshev',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Chebyshev')



# Chi-Square

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='chi2',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Chi Square')




# Braycurtis

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='braycurtis',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Bray Curtis')




# Canberra

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='canberra',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Canberra')



# Cosine

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='cosine',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Cosine')


# Correlation

rank_accuracies, mAP = evaluate_metric(X_test, y_test, X_test, y_test,
                                       metric ='correlation',
                                       parameters = None)

rank_accuracies_l.append(rank_accuracies)
mAP_l.append(mAP)
metric_l.append('Correlation')


plt.figure(figsize=(8.0, 6.0))
color_list = ['green', 'blue', 'red', 'purple', 'orange', 'magenta', 'cyan', 'black', 'indianred', 'lightseagreen', 'gold', 'lightgreen']
for i in range(len(metric_l)):
    plt.plot(np.arange(1, 31), 100*rank_accuracies_l[i], color=color_list[i], linestyle='dashed', label='Metric: '+ metric_l[i])



# plt.title('CMC Curves for a range of standard distance metrics')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics.png', dpi=1000, transparent=True)
#
# plt.title('CMC Curves for a range of standard distance metrics (L2 norm)')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics_L2_norm.png', dpi=1000, transparent=True)

# plt.title('CMC Curves for a range of standard distance metrics (histogram-10)')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics_hist_10.png', dpi=1000, transparent=True)
# # #
# plt.title('CMC Curves for a range of standard distance metrics (histogram-51)')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics_hist_51.png', dpi=1000, transparent=True)

# plt.title('CMC Curves for a range of standard distance metrics (histogram-255)')
# plt.xlabel('Rank')
# plt.ylabel('Recogniton Accuracy [%]')
# plt.legend(loc='best')
# plt.savefig('img/CMC_curves_for_distance_metrics_hist_256.png', dpi=1000, transparent=True)


plt.title('CMC Curves for a range of standard distance metrics (HOG)')
plt.xlabel('Rank')
plt.ylabel('Recogniton Accuracy [%]')
plt.legend(loc='best')
plt.savefig('img/CMC_curves_for_distance_metrics_HOG.png', dpi=1000, transparent=True)


# import matplotlib.pyplot as plt
#
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=plt.figaspect(0.35))
#
# ax = fig.add_subplot(1, 2, 1)
#
# #fig = plt.figure(1, figsize=(8, 6))
# #ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(X_test.T)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_test.ravel(),
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA dimensions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# # ax.set_zlabel("3rd eigenvector")
# # ax.w_zaxis.set_ticklabels([])
#
# ax = fig.add_subplot(1, 2, 2, projection='2d')
#
# #ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(X_test_norm.T)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_test.ravel(),
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA dimensions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# # ax.set_zlabel("3rd eigenvector")
# # ax.w_zaxis.set_ticklabels([])
#
#
# plt.show()