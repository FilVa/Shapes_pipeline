# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:33:18 2019

@author: filipavaldeira
"""

import numpy as np
import open3d as o3d

def get_flat_ids(ids_vec,dim,n_points):
    # input is a vector of ids and returns the correspondent flat ids for a mtrix with columns dim

    n_ids = np.array(ids_vec).reshape(-1,1).shape[0]
    arr = np.zeros((2,n_ids*dim))
    arr[0,:] = np.repeat(ids_vec, dim)
    if(dim==3): 
        arr[1,:] = np.repeat([[0,1,2]], n_ids,axis=0).reshape(1,-1)[0]
    elif(dim==2):
        arr[1,:] = np.repeat([[0,1]], n_ids,axis=0).reshape(1,-1)[0]

    arr_dim = (int(n_points),int(dim))

    final = np.ravel_multi_index(arr.astype(int), arr_dim)

    return final
    

def array_filter_id_nan(arr,ids):
    if(len(arr.shape)==1):
        new_arr = np.empty((len(ids)))
    else:
        new_arr = np.empty((len(ids),arr.shape[1]))
        
    new_arr[:]  = np.nan
    ids_total = np.arange(len(ids))    
    mask = np.isnan(ids)    
    
    new_arr[ids_total[~mask].astype(int)] = arr[ids[~mask].astype(int)]
    return new_arr

def convert_to_dict(keys,values):
    dictionary = dict(zip(keys, values))
    return dictionary    

# gets arr with ids are subset of total ids from 0 to sz and returns the non present ids
def find_complementary_id(arr,sz):
    
    total_id = np.arange(sz)
    complement = [i for i in total_id if not np.any(i==arr)]
    
    complement_arr = np.array(complement)
    return complement_arr

# Correspondences handlers
def correspondence_switch(arr,sz_other):
    # arr is the array with id correspondences
    # sz_other is the size of the other array in order to know which ids do not have correspondences
    
    other_arr = np.empty(sz_other)
    other_arr[:]  = np.nan

    arr_id = np.arange(len(arr))
    non_nan_arr = arr[~np.isnan(arr)]
    non_nan_ids = arr_id[~np.isnan(arr)]
    
    other_arr[non_nan_arr.astype('int')] = non_nan_ids    
    
    return other_arr


# Flat matrix to 3d matrix. 3rd dimension for different samples
def flat2mat(flat_data,k_landmarks,dim):
    
    if(flat_data.shape[0]==1): # only one sample
        mat_data = np.zeros((k_landmarks,dim))
        mat_data = np.reshape(flat_data,(k_landmarks,dim))
    else:
        mat_data = np.zeros((k_landmarks,dim,flat_data.shape[0]))    
        for row in range(0,flat_data.shape[0]):
            mat_data[:,:,row] = np.reshape(flat_data[row,:],(k_landmarks,dim))
    return mat_data

# 3d matrix OR 2d list, to 2d matrix. Each row corresponds to one sample(shape)
def mat2flat(mat_data):
    # input is list
    if(isinstance(mat_data,list)):
        i = 0
        sz = mat_data[0].shape
        flat_data = np.zeros((len(mat_data),sz[0]*sz[1]))
        for sample in mat_data:
            flat_data[i,:] = np.reshape(sample,(1,-1))
            i += 1
    # input is matrix
    else:
        if(len(mat_data.shape) == 2): # only one sample
            mat_data = np.reshape(mat_data,(mat_data.shape[0],mat_data.shape[1],1))
        sz = mat_data.shape
        flat_data = np.zeros((sz[2],sz[0]*sz[1]))
        for sample in range(0,sz[2]):
            flat_data[sample,:] = np.reshape(mat_data[:,:,sample],(1,-1))
    return flat_data

def check_matrix(matrix,dim,k_landmarks,n_samples,goal):
    
    sz = matrix.shape
    # Flat matrix : should have shape n_samples*(dim*k_landmarks)
    if(len(sz)==2):
        if((sz[0]==n_samples) and (sz[1] == dim*k_landmarks)):
            if(goal=='flat'): #want flat no need to transform
                return matrix
            elif(goal=='3d'): # transform flat to 3D
                matrix = flat2mat(matrix,k_landmarks,dim)
                return matrix                
        else:
            print('Data matrix has non coherent dimensions')
            return None
    
    # 3D matrix : should have shape k_landmarks*dim*n_samples
    elif(len(sz)==3):
        if((sz[0]==k_landmarks) and (sz[1] == dim) and (sz[2]==n_samples)):
            if(goal=='3d'):
                return matrix
            elif(goal=='flat'):
                matrix=mat2flat(matrix)
                return matrix                     
        else:
            print('Data matrix has non coherent dimensions')
            return None
   
# Get numpy array and return PointCloud format required by open3d framework
def numpy2PointCloud(coor):
    if(coor.shape[1]==2):
        stub_zeros = np.zeros((coor.shape[0],1))
        points = np.concatenate((coor,stub_zeros),axis=1)
    else:
        points = coor
            
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud  


