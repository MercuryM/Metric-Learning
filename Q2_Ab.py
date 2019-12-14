# -*- coding: utf-8 -*-
"""
Spyder Editor
This is agglomerative clustering
This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
#import matplotlib.animation as animation
from Split import split_data
from scipy.optimize import linear_sum_assignment
#
#k = 20

def euclidian(a, b):
    return np.linalg.norm(a-b)

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
f = time.time()-s
cost = cost_matrix(label)        
row_ind, col_ind = linear_sum_assignment(cost)
cost_sum=cost[row_ind, col_ind].sum()
accuracy = cal_accuracy(32,cost_sum)