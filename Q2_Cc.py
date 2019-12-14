# -*- coding: utf-8 -*-
"""
Spyder Editor
put 4 representations together

This is a temporary script file.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from numpy import *
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from Split import split_data
from scipy.optimize import linear_sum_assignment
from Evaluate import get_acc_score, evaluate_metric

def euclidian(a, b):
    c = np.linalg.norm(a-b)
    return c


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


#calculate distance vector
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
    for m in range(200):
        for n in range(l1):
            feature_vec[m,n] = euclidian(test_X[:,m], mean_mat_1[:,n])  
    return {'mean_mat': mean_mat_1, 'feature_vector': feature_vec}

#for self-defined gmm

def PCA_proj(train,test,M):
    
    pca = PCA(n_components=M)
    pca.fit(train.T)
    
    new_train = pca.transform(train.T)
    new_test = pca.transform(test.T)
    
    return {'train':new_train.T,'test': new_test.T}
    

#get parameter for gaussian
def gaussian_para(record, train, test):
    D,N = train.shape
    X = train.T
    
    K = len(record)
    mean_collect = []
    cov_collect = []
    pi_collect = []
    
    for k in range(K):
        
        mean = np.reshape(np.mean(X[record[k]], axis=0),(1,D))
        A = X[record[k]]-mean  
        N_k = len(record[k])
        cov = (1/N_k)*((A.T).dot(A)) #2576*2576
        for i in range(2):
            cov[i,i] = cov[i,i]+0.00000001
        pi = N_k/N
        
        mean_collect.append(mean)
        cov_collect.append(cov)
        pi_collect.append(pi)
        
    mean_mat = np.reshape(mean_collect,(K,D))
    mean_mat = mean_mat.T
    
    return {'mean_mat': mean_mat, 'cov': cov_collect,'pi': pi_collect}

 #get parameter for fisher vector
def get_para(record, train, test):
    D,N = train.shape
    X = train.T
    
    K = len(record)
    mean_collect = []
    cov_collect = []
    pi_collect = []
    
    for k in range(K):
        
        mean = np.reshape(np.mean(X[record[k]], axis=0),(1,D))
        A = X[record[k]]-mean  
        N_k = len(record[k])
        cov = (1/N_k)*((A.T).dot(A))
        cov=np.diag(cov)

        cov = cov + 0.00000001
        pi = N_k/N
        
        mean_collect.append(mean)
        cov_collect.append(cov)
        pi_collect.append(pi)

    mean_mat = np.reshape(mean_collect,(K,D))
    mean_mat = mean_mat.T
    
    return {'mean_mat': mean_mat, 'cov': cov_collect,'pi': pi_collect}
def gaussian_prob(x, mu_k, cov_k):
    
    norm = multivariate_normal(mean=mu_k,cov=cov_k)
    return norm.pdf(x)
        
    
def gaussion_mix(record,test,mean,cov,wk):
    K = len(record)
    N = len(test[0,:]) #D,N
    X = test.T
#    wk = wk.T #k*n->n*k
    gamma = np.full((N,K),0.)
    prob = np.full((N,K),0.)
    for k in range(K):
        prob[:,k] = gaussian_prob(X, mean[:,k], cov[k])
    for k in range(K):
        gamma[:, k] = wk[k]*prob[:,k]
    return gamma.T, prob.T


#calculate fisher vectors
def fisher_vector(record,test,mu,sigma,wk,gamma):
    
    D, N = test.shape
    K = len(record)  #num of clusters

    
    f_vector = np.full((2*K*D,1),0.)
    f_matrix = np.full((2*K*D,N),0.)
    for n in range(N):

        for k in range(K):

            w = wk[k]
            r = gamma[k,n]
            X = test[:,n]
            u = mu[:,k]
            cov = sigma[k]
            v_k = (1/np.sqrt(w))*r*(X-u)/cov
            if np.linalg.norm(v_k)!= 0:
                v_k = v_k/np.linalg.norm(v_k)
            u_k = (1/np.sqrt(2*w))*r*((X-u)/cov - 1)**2
            if np.linalg.norm(u_k)!= 0:
                u_k = u_k/np.linalg.norm(u_k)
            fv = np.hstack((v_k,u_k))
            fv = np.reshape(fv,(2*D*1,1))
            f_vector[2*D*k:2*D*(k+1)] = fv
        f_vector = f_vector/np.linalg.norm(f_vector) 

        f_matrix[:,n:(n+1)] = f_vector
        
    return f_matrix
        
    
t = time.time()

data = split_data()
train_X, train_l = data['train']
D_train, N_train = train_X.shape
test_X, test_l = data['test']
D_test, N_test = test_X.shape    

#l2 norm
train_X = train_X/np.apply_along_axis(np.linalg.norm, 0, train_X)
test_X = test_X/np.apply_along_axis(np.linalg.norm, 0, test_X)


label = agglomerative(32,train_X)


feature = get_feature(label, train_X, test_X)
mean_mat_1 = feature['mean_mat']

feature_vec = feature['feature_vector'].T
feature_vec_inv = 1./feature_vec
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

softmax_p = softmax(feature_vec_inv)


#for the first gmm
rank1_g1 = []
rank10_g1 = []
mAP_g1 = []
#for the second gmm
rank1_g2 = []
rank10_g2 = []
mAP_g2 = []
#for count
count = []

for i in range(17):
    
    print(i)
    #divide clusters
    label = agglomerative((i+1)**2,train_X)
    #get feature vectors
    feature = get_feature(label, train_X, test_X)
    ini_mean = feature['mean_mat']
    feature_vec = feature['feature_vector'].T
    

    #first gmm
    proj_result = PCA_proj(train_X,test_X,2)

    train = proj_result['train']
    test = proj_result['test']
    
    para = gaussian_para(label, train, test)
    mean_mat = para['mean_mat']
    cov = para['cov']
    pi = para['pi']


    fisher_para = get_para(label, train_X, test_X)
    f_mean_mat = fisher_para['mean_mat']
    f_cov = fisher_para['cov']


    gamma,prob = gaussion_mix(label,test,mean_mat,cov, pi)

    fisher_vectors = fisher_vector(label,test_X,f_mean_mat,f_cov,pi,gamma)
    rank_accuracies, mAP = evaluate_metric(fisher_vectors, test_l, fisher_vectors, test_l,
                                       metric ='euclidian',
                                       parameters = None)
    rank1_3 = rank_accuracies[0]
    rank10_3 = rank_accuracies[9]
    mAP_3 = mAP

    rank1_g1.append(rank1_3)
    rank10_g1.append(rank10_3)
    mAP_g1.append(mAP_3)
    
    #second gmm
    X= train_X.T
    ini_mean = ini_mean.T
    gmm = GaussianMixture(n_components = (i+1)**2,covariance_type = 'diag', means_init = ini_mean, random_state = 10)
    gmm.fit(X)
    gmm_weights = gmm.weights_
    gmm_centres = gmm.means_
    gmm_cov = gmm.covariances_
    gmm_prob = gmm.predict_proba(test_X.T)
    gmm_prob = gmm_prob.T

    
    fisher_vectors = fisher_vector(label,test_X,gmm_centres.T,gmm_cov,gmm_weights,gmm_prob)
    rank_accuracies, mAP = evaluate_metric(fisher_vectors, test_l, fisher_vectors, test_l,
                                       metric ='euclidian',
                                       parameters = None)
    rank1_4 = rank_accuracies[0]
    rank10_4 = rank_accuracies[9]
    mAP_4 = mAP
    rank1_g2.append(rank1_4)
    rank10_g2.append(rank10_4)
    mAP_g2.append(mAP_4)
    
    count.append((i+1)**2)
    
x = count

y3_1 = rank1_g1
y3_2 = rank10_g1
y3_3 = mAP_g1

y4_1 = rank1_g2
y4_2 = rank10_g2
y4_3 = mAP_g2

plt.figure(figsize=(8,6))

plt.plot(x,y3_1,  color='steelblue', marker='o',linewidth = 2,label = '@rank1(1st GMM)')
plt.plot(x,y4_1,  color='steelblue', linestyle='dashed', marker='*',linewidth = 2,label = '@rank1(2nd GMM)')
plt.plot(x,y3_2,  color='darkorange', marker='o',linewidth = 2,label = '@rank10(1st GMM)')
plt.plot(x,y4_2,  color='darkorange', linestyle='dashed', marker='*',linewidth = 2,label = '@rank10(2nd GMM)')
plt.plot(x,y3_3,  color='forestgreen', marker='o',linewidth = 2,label = 'mAP(1st GMM)')
plt.plot(x,y4_3,  color='forestgreen', linestyle='dashed', marker='*',linewidth = 2,label = 'mAP(2nd GMM)')

plt.legend()
x_ticks = np.arange(0,290,30)
y_ticks = np.arange(0.0,1.0,0.1)
plt.xlim((0.0, 290))
plt.ylim((0.0, 1.0))
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('Cluster Number N')
plt.ylabel('KNN Accuracies or mAP')
plt.title('KNN results for two GMMs')  
plt.savefig('img/KNN results for two GMMs.png',
                 dpi=1000, transparent=True)


t_F = time.time()-t