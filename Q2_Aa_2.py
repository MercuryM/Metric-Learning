# -*- coding: utf-8 -*-
"""
Spyder Editor
This is the kmeans using packages
This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Split import split_data
from scipy.optimize import linear_sum_assignment


def euclidian(a, b):
    return np.linalg.norm(a-b)


def cost_matrix(record):
    k = len(record)
    cost = np.full((k,k),0)
    for i in range(k):
        cluster = record[i]
        for j in range(len(cluster)):
            m = cluster[j]
            n = train_l[0,m]
            cost[i,:] = cost[i,:] + 1
            cost[i,n-1] = cost[i,n-1] - 1
    return cost

def find_clurster_mean(k,times,random_num,test):
#    X = test.T    
    all_accuracy = []
    time_p = []
    np.random.seed(random_num)
    for i in range(times):
        r = np.random.randint(0,1000)
#        t = time.time()
        label,f = k_means(k,r,test)
#        f = time.time() - t
        cost = cost_matrix(label)        
        row_ind, col_ind = linear_sum_assignment(cost)
        cost_sum=cost[row_ind, col_ind].sum()
        accuracy_1 = cal_accuracy(k,cost_sum)
        time_p.append(f)
        all_accuracy.append(accuracy_1)
    average_t = np.mean(time_p)
    avg_accuracy = np.mean(all_accuracy)
    return avg_accuracy, average_t

def cal_accuracy(k,cost_sum):
    return (10*k-cost_sum)/(10*k)
    
def k_means(k,state,test):
    X = test.T
    s =  time.time()
    kmeans = KMeans(n_clusters=k, random_state=state).fit(X)
    f = time.time()-s
    clur_label = kmeans.labels_
    label = [] 
    for i in range(k):
        temp = []
        for j in range(len(clur_label)):
            if i == clur_label[j]:
                temp.append(j)
        label.append(temp)
    return label,f

data = split_data()
train_X, train_l = data['train']
D_train, N_train = train_X.shape
test_X, test_l = data['test']
D_test, N_test = test_X.shape    
#l2 norm
train_X = train_X/np.apply_along_axis(np.linalg.norm, 0, train_X)
test_X = test_X/np.apply_along_axis(np.linalg.norm, 0, test_X)
  
accuracy,t = find_clurster_mean(32,1,10,train_X)
#ft = time.time()-s
