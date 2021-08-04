# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:39:40 2020

@author: filipavaldeira
"""

import gpflow
import tensorflow as tf
import numpy as np
from utils.convert import get_flat_ids
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

def tf_kron(a,b):
    a_shape = [a.shape[0],a.shape[1]]
    b_shape = [b.shape[0],b.shape[1]]
    return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])

def tf_isin(a,b):
    a0 = tf.expand_dims(a, 1)
    b0 = tf.expand_dims(b, 0)
    return tf.reduce_any(tf.equal(a0, b0), 1)


# ------------- KERNELS

# Class to merge together the PCA kernel with the SE kernel
class TotalKernel(gpflow.kernels.Kernel):
    def __init__(self,kernel_list,dim):
        super().__init__()
        if hasattr(kernel_list, 'kernels'):  
            self.kernel_list = kernel_list.kernels     
        else:            
            self.kernel_list =list()
            self.kernel_list.append(kernel_list)
        self.dim = dim
        
        for kernel in self.kernel_list:
            if(kernel.name=='empirical_kernel'):
                self.full_pca_cov = kernel.cov                                  
        
    def K(self, X, X2=None, corr_vec=None): 
        if X2 is None:
            X2 = X
            
        full_cov = np.zeros((X.shape[0],X2.shape[0]))
        
        for kernel in self.kernel_list:
            if(kernel.name=='empirical_kernel'):
                cov = kernel.K(X,X2)                               
            else:
                X_points = tf.reshape(X,(-1,self.dim))
                X2_points = tf.reshape(X2,(-1,self.dim))
                og_k = kernel.K(X_points,X2_points)
                cov =tf_kron(og_k, tf.eye(self.dim,dtype=tf.dtypes.double))                
            full_cov = full_cov + cov
        return full_cov
                    
            
    def K_diag(self, X):
        full_cov = np.zeros((X.shape[0]))
        for kernel in self.kernel_list:
            if(kernel.name=='empirical_kernel'):
                cov = kernel.K_diag(X)                                
            else:
                X_points = tf.reshape(X,(-1,self.dim))
                og_k = kernel.K_diag(X_points)
                cov = tf.repeat(og_k,self.dim)
                
            full_cov = full_cov + cov
        return tf.constant(full_cov, tf.float64)
        

# Empirical kernel implementation
class EmpiricalKernel(gpflow.kernels.Kernel):
    # this is not the old empirical_kernel 
    def __init__(self,ref_shape,shape_list,dim,n_components):
        super().__init__()
        
        self.ref_shape = ref_shape
        self.dim = dim
        self.shape_list = shape_list
        self.n_comp = n_components
        
        self.n_train_samples = len(shape_list)
        self.n_points = ref_shape.shape[0]
        
        self.mean_def,self.train_mat = self.get_train_mat()
        self.cov = self.get_covariance()
        
        self.ref_tree = KDTree(ref_shape)
        self.mean_function = Mean_pca(ref_shape, dim, self.mean_def)               
        
    def get_train_mat(self):
        
        i=0
        train_mat = np.zeros((self.n_train_samples,self.n_points*self.dim))
        for shape in self.shape_list:
            train_mat[i,:] = (shape-self.ref_shape).reshape(1,-1)[0,:]
            i += 1
            
        mean_def = train_mat.mean(axis=0)
        #train_mat = train_mat - mean_shape
        
        return mean_def,train_mat

    
    def get_covariance(self):

        self.pca = PCA(n_components=self.n_comp)
        train_data = self.train_mat
        pca_result = self.pca.fit_transform(train_data)
        cov = self.pca.get_covariance()
        return cov
    

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        # Turn into points
        X_points = tf.reshape(X,(-1,self.dim))
        X2_points = tf.reshape(X2,(-1,self.dim))
       # Find closest ids
        X_dist,X_ids = self.ref_tree.query(X_points)

        if(X_ids.shape[0]==0):
            # No points (usually because we are sampling from posterior). choose first point of ref shape
            X_ids = tf.where(tf_isin(self.ref_shape,self.ref_shape[0,:])[:,0])

        X2_dist,X2_ids = self.ref_tree.query(X2_points)
        if(X2_ids.shape[0]==0):
            X2_ids = tf.where(tf_isin(self.ref_shape,self.ref_shape[0,:])[:,0])
 
        X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
        X2_ids_flat = get_flat_ids(X2_ids,self.dim,self.n_points)
        
        filter_X = self.cov[X_ids_flat,:]
        filter_X2 = filter_X[:,X2_ids_flat]
        
        final_mat = filter_X2
        cov_mat_tf = tf.constant(final_mat, tf.float64)  

        return cov_mat_tf
        

    def K_diag(self, X):
        
        X_points = np.reshape(X,(-1,self.dim))
        X_dist,X_ids = self.ref_tree.query(X_points)
        X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
        diag = self.cov.diagonal()        
        cov_mat_tf = tf.constant(diag[X_ids_flat], tf.float64)

        return cov_mat_tf 
    
class Mean_pca(gpflow.mean_functions.MeanFunction):

    def __init__(self,ref_shape,dim,mean_def):
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.ref_tree = KDTree(ref_shape.copy())
        self.u_mean = mean_def.reshape(-1,1).copy()
        self.dim  = dim
        self.n_points = ref_shape.shape[0]

    def __call__(self, X,corr_vec=None):
        
        X_points = np.reshape(X,(-1,self.dim))
        X_dist,X_ids = self.ref_tree.query(X_points)
        X_ids_flat = get_flat_ids(X_ids,self.dim,self.n_points)
        mean_vec = self.u_mean[X_ids_flat,:]
        
        return tf.constant(mean_vec.reshape(-1,1), tf.float64)    