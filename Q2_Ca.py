# -*- coding: utf-8 -*-
"""
Spyder Editor
This is the distance vector representation
This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from Split import split_data
from scipy.optimize import linear_sum_assignment
from Evaluate import get_acc_score, evaluate_metric


def euclidian(a, b):
    c = np.linalg.norm(a-b)
    return c


def cost_matrix(record):
    k = len(record)
    cost = np.full((k,k),0)
    for i in range(32):
        cluster = record[i]
        for j in range(len(cluster)):
            m = cluster[j]
            n = train_l[0,m]
            cost[i,:] = cost[i,:] + 1
            cost[i,n-1] = cost[i,n-1] - 1
    return cost

def cal_accuracy(k,cost_sum):
    return (10*k-cost_sum)/(10*k)
    
def agglomerative(k,test):
    X = test.T
#    clustering = AgglomerativeClustering(n_clusters=k,compute_full_tree=True).fit(X)
    clustering = AgglomerativeClustering(n_clusters=k).fit(X)
    clur_label = clustering.labels_
    label = [] 
    for i in range(k):
        temp = []
        for j in range(len(clur_label)):
            if i == clur_label[j]:
                temp.append(j)
        label.append(temp)
    return label



def get_feature(record, train, test):
    X = train.T
    n1,n2 = test.shape
    
    l1 = len(record)
    mean_collect = []
    for i in range(l1):
        mean = np.reshape(np.mean(X[record[i]], axis=0),(1,2576))
        mean_collect.append(mean)
    mean_mat_1 = np.reshape(mean_collect,(l1,2576))
    mean_mat_1 = mean_mat_1.T
    
    feature_vec = np.full((n2,l1),0.)
    for m in range(n2):
        for n in range(l1):
            feature_vec[m,n] = euclidian(test_X[:,m], mean_mat_1[:,n])  
    return {'mean_mat': mean_mat_1, 'feature_vector': feature_vec}

data = split_data()
train_X, train_l = data['train']
D_train, N_train = train_X.shape
test_X, test_l = data['test']
D_test, N_test = test_X.shape    
#l2 norm
train_X = train_X/np.apply_along_axis(np.linalg.norm, 0, train_X)
test_X = test_X/np.apply_along_axis(np.linalg.norm, 0, test_X)

s = time.time()
label = agglomerative(32,train_X)



feature = get_feature(label, train_X, test_X)
mean_mat = feature['mean_mat']
mean_mat = mean_mat/np.apply_along_axis(np.linalg.norm, 0, mean_mat)
feature_vec = feature['feature_vector'].T

rank_accuracies, mAP = evaluate_metric(feature_vec, test_l, feature_vec, test_l,
                                       metric ='euclidian',
                                       parameters = None)
f = time.time()-s
a = np.linalg.norm(mean_mat[:,2])
rank1 = []
rank10 = []
ac_mAP = []
count = []
for i in range(17):
    label = agglomerative((i+1)**2,train_X)
    feature = get_feature(label, train_X, test_X)
    mean_mat = feature['mean_mat']
    mean_mat = mean_mat/np.apply_along_axis(np.linalg.norm, 0, mean_mat)
    feature_vec = feature['feature_vector'].T

    rank_accuracies, mAP = evaluate_metric(feature_vec, test_l, feature_vec, test_l,
                                           metric ='euclidian',
                                           parameters = None)
    rank1.append(rank_accuracies[0])
    rank10.append(rank_accuracies[9])
    ac_mAP.append(mAP)
    count.append((i+1)**2)
    
x1 = count
y1 = rank1
y2 = rank10
y3 = ac_mAP

plt.figure(figsize=(8,6))
plt.plot(x1,y1,  marker='o',linewidth = 2,label = '@rank1')
plt.plot(x1,y2,  marker='*',linewidth = 2,label = '@rank10')
plt.plot(x1,y3,  marker='x',linewidth = 2,label = 'mAP')


plt.legend()
x_ticks = np.arange(0,290,30)
y_ticks = np.arange(0.0,1.0,0.1)
plt.xlim((0.0, 290))
plt.ylim((0.0, 1.0))
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('Cluster Number N')
plt.ylabel('KNN Accuracies')
plt.title('KNN Results of Distance Vector Representation versus Cluster Numbers')  
plt.savefig('img/KNN Accuracies of Distance Vector Representation versus Cluster Numbers.png',
                 dpi=1000, transparent=True)
