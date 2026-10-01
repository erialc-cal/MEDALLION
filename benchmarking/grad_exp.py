import numpy as np
from sklearn.metrics import pairwise_distances
import math
from scipy.spatial.distance import squareform
from tqdm import tqdm
import matplotlib.pyplot as plt

def calc_direct_2order(tsne, feat, idx):
    p=squareform(tsne.P_)  #n x n
    q=squareform(tsne.Q_)  #n x n
    v=p-q        #symmetric
    n=feat.shape[0]
    m=feat.shape[1]

    d=feat[idx]-feat   #n x m
    e=1+np.linalg.norm(d, axis=1)**2 #n x 1
    d_=np.identity(m)  #n x n
    e_=2*d             #n x m

    total_e=1+np.sum(feat[:,None,:]-feat[None,:,:]**2, axis=2)
    sq=0              #scalar
    for i in range(n):
        for j in range(i+1,n):
            sq+=1/total_e[i,j]
    sq*=2   #definitive
    sq_=np.zeros((1,m))  #definitive
    for i in range(n):
        sq_+=1/e[i]*d[i]
    sq_*=-4

    v_=np.zeros((n,m))
    for i in range(n):
        v_[i]=2/e[i]**2*sq*d[i] + 1/e[i]*sq_
    v_/=sq**2

    direct_2order=np.zeros((m,m))
    for i in range(n):
        direct_2order+=1/e[i]* (np.dot(v_[i].T,d[i]) + v[idx,i]*d_ - v[idx,i]/e[i]*np.dot(e_[i].T,d[i]))

    return direct_2order

def calc_cross2order_single(tsne, x_train, feat, idx, k):
    scales=tsne.scales_      ##this is 1/(2*sigma**2)
    d=feat[idx]-feat    ##y_i - y_j   #n x m
    e=1+np.linalg.norm(d, axis=1)**2  #n x 1
    n=feat.shape[0]
    m=feat.shape[1]

    x_sq_distances_tot=pairwise_distances(x_train, squared=True)
    x_sq_distances=x_sq_distances_tot[idx] #n x 1
    
    x_k=x_train[:,k]      #n x 1
    x_k_distances=np.abs(x_k[:, np.newaxis]-x_k)
    
    s_pi=0   #eq 21
    partial_s_pi=0  #eq23
    for l in range(n):
        if idx!=l:
            temp=math.exp(-x_sq_distances[l]/scales[idx])
            s_pi+=temp
            partial_s_pi-=x_k_distances[idx,l]*2*scales[idx]*temp

    cross_2order=np.zeros((1,m))
    for j in range(n):   #j 
        e_ij=e[j]
        d_ij=d[j]
        v_ij_=0

        p_ji=math.exp(-x_sq_distances[j]/scales[idx])/s_pi   #eq20
        partial_p_ji=-(x_k_distances[idx,j]*2*scales[idx] + partial_s_pi/s_pi)*p_ji  #eq22

        s_pj=0    #eq25
        for l in range(n):
            if j!=l:
                s_pj+=math.exp(-x_sq_distances_tot[j][l]/scales[j]) 
            
        temp=math.exp(-x_sq_distances[j]/scales[j])
        p_ij=temp/s_pj  #eq 24
        partial_s_pj=x_k_distances[j,idx]*2*scales[j]*temp    #eq27
        partial_p_ij=(x_k_distances[j,idx]*2*scales[j] + partial_s_pj/s_pj)*p_ij #eq26

        v_ij_=partial_p_ij + partial_p_ji

        cross_2order+=v_ij_/e[j]*d[j]
        
    cross_2order*=2/n
    return cross_2order

def cross_2order_total(tsne, x_train, feat, idx):
    #for equation 6
    low_dim=feat.shape[1]
    high_dim=x_train.shape[1]
    
    total=np.empty((low_dim,high_dim))
    for k in (range(high_dim)):
        cross=calc_cross2order_single(tsne, x_train, feat, idx, k)
        total[:,k]=cross

    return total


def grad_wtr_y(tsne, feat):
    ##equation 7
    p=squareform(tsne.P_)
    q=squareform(tsne.Q_)
    n=feat.shape[0]
    low_dim=feat.shape[1]

    grad=np.zeros((n,low_dim))
    for i in tqdm(range(n)):
        temp=0
        d=feat[i]-feat    
        e=1+np.linalg.norm(d, axis=1)**2
        for j in range(n):
            if i!=j:
                temp+=(p[i,j]-q[i,j])*d[j]/(e[j]) 
        temp*=4
        grad[i]=temp

    return grad


def grad_wtr_x(direct_2order, cross_2order_total):
    ##equation 6
    inv=np.linalg.inv(direct_2order)
    grad=-np.dot(inv, cross_2order_total)
    return grad

