# -*- coding: utf-8 -*-
"""
Spyder Editor
This is the self-defined k-means
This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from Split import split_data
from scipy.optimize import linear_sum_assignment


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

def find_clurster_mean(k,times,random_num,test):
#    X = test.T
    all_accuracy = []
    np.random.seed(random_num)
    for i in range(times):
        cluster = kmeans(k,test)
        label = cluster['cluster_elements']
        cost = cost_matrix(label)        
        row_ind, col_ind = linear_sum_assignment(cost)
        cost_sum=cost[row_ind, col_ind].sum()
        accuracy_1 = cal_accuracy(k,cost_sum)
        all_accuracy.append(accuracy_1)
    avg_accuracy = np.mean(all_accuracy)
    return avg_accuracy
    
def cal_accuracy(k,cost_sum):
    return (10*k-cost_sum)/(10*k)


def kmeans(k, test,epsilon=0, distance='euclidian'):
    #store the past centroid
    history_centroids = []
    #method to calculate the distance
    if distance == 'euclidian':
        dist_method = euclidian

    data_X = test.T
    N_test, D_test = data_X.shape
    
    #randomly choosing 20 point as the inital centroids
    np.random.seed(1)
    arr = []
    while len(arr) < k:
        r = np.random.randint(0,N_test)
        if r not in arr: arr.append(r)
    
    #arr 里面储存的行数
    mean_X = data_X[arr]
    
    #store the means
    history_centroids.append(mean_X.T)
    
    mean_X_pre = np.full(mean_X.shape,0.)
    
    #store the clusters
    #all the data points contained in the clusters
    belongs_to = np.full((N_test,1),0.)
    norm = dist_method(mean_X, mean_X_pre)
    
    
    iteration = 0
    
    #epsilon = 0 is the stop condition
    while norm > epsilon:
        iteration += 1
        
        norm = dist_method(mean_X, mean_X_pre)
        mean_X_pre = mean_X
        
        #for each face in the dataset
        for index_image, image in enumerate(data_X):
            #储存每一个点距离所有mean的距离
            dist_vec = np.full((k,1),0)
            #for each mean/centroid
            for index_mean, mean in enumerate(mean_X):
                #compute the distance between samples and centroid
                dist_vec[index_mean] = dist_method(mean, image)
            #储存每一个点距离最近的mean
            belongs_to[index_image, 0] = np.argmin(dist_vec)
        
        tmp_prototypes = np.full((k,D_test),0)
        
        #k 个cluster mean 找belongs_to中属于他们的data
        label_cluster = []
        #len(belongs_to == 10*k)
        for index in range(k):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            #this is to calculate the mean
            #new centroid
            prototype = np.mean(data_X[instances_close], axis=0)
            
            #new centroid set
            tmp_prototypes[index, :] = prototype
            label_cluster.append(instances_close)
        mean_X = tmp_prototypes

        history_centroids.append(tmp_prototypes.T)
    return {'cluster_means': mean_X.T, 'initial_label': belongs_to, 'iteration_num': iteration, 'cluster_elements': label_cluster}   

data = split_data()
train_X, train_l = data['train']
D_train, N_train = train_X.shape
test_X, test_l = data['test']
#l2 norm
train_X = train_X/np.apply_along_axis(np.linalg.norm, 0, train_X)
test_X = test_X/np.apply_along_axis(np.linalg.norm, 0, test_X)

D_test, N_test = test_X.shape    
avg_accuracy = find_clurster_mean(32,10,10,train_X)

    
