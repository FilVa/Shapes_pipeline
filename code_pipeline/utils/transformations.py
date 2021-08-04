# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:52:36 2019

@author: filipavaldeira
"""

import numpy as np
import math
from scipy.linalg import orthogonal_procrustes
from utils.convert import mat2flat

# Rotate P into Q
# Return rotated P 'solution' and rotation matrix 'R'
def OPA(Q,P) :
      res =orthogonal_procrustes(P,Q);
      R = res[0]
      solution = np.matmul(P, R);
      return solution, R

# Cost function of GPA
def getG(Xp,m,k,n):
    
    Xps = np.reshape(Xp,(m*k,n));
    G = 0;
    for i in range(n) :
        rept = np.tile(Xps[:,i],(n-i-1,1));
        others = Xps[:,i+1:n];
        diff = rept - np.transpose(others);
        squared = np.square(diff);
        G = G+ np.sum(squared);
    G = G/n;   
    return G
    
# Generalized Procrustes Analysis         
# Receives k*m*n matrix 'shapes' : k is number of landmarks, m is dimension, n is number of samples
# Applies GPA with scaling if scale_flag =1
def GPA(shapes, scale_flag, thresh_tol):
    print('Start Generalized Procrustes Analysis')
    [n_points,dim,n_samples] = shapes.shape
    
    transformations = np.zeros((dim+1,dim+1,n_samples))
    X_bar= np.zeros((n_points,dim))
    X_p = np.zeros((n_points,dim,n_samples))
    
    # Translations
    print('Translation step')
    C= np.identity(n_points)-(1/n_points)*(np.ones((n_points,1))*np.ones((1,n_points))) 
    for i in range(n_samples):
        X_p[:,:,i] = np.matmul(C,shapes[:,:,i])
        diff =  X_p[:,:,i] - shapes[:,:,i]
        transformations[0:3,3,i] = diff[0,:]
                    
    for i in range(n_samples):
        transformations[0:3,0:3,i] = np.identity(dim)
        transformations[3,3,i] = 1
    
    # Initialize tolerance
    tol_old = 1e16
    tol_new = 1e15
    while (abs(tol_old - tol_new) > thresh_tol):
        
        # Rotation cycle
        print('Rotation cycle')
        while (abs(tol_old - tol_new) > thresh_tol):            
            tol_old = tol_new;            
            for i in range(n_samples):                
                # exclude sample from mean
                before_sample = X_p[:,:,0:i]
                after_samples = X_p[:,:,i+1:n_samples]
                join =  np.zeros((n_points,dim,n_samples-1))
                join[:,:,0:i] = before_sample
                join[:,:,i:n_samples] = after_samples
                
                X_bar = np.mean(join,2) # Mean  shape
                X_p[:,:,i],rotation = OPA(X_bar, X_p[:,:,i])
                # Save rotation matrix
                transformations[0:3,0:3,i] = np.matmul(transformations[0:3,0:3,i],np.transpose(rotation)) 
                # Update translation
                transl_rot= np.matmul(np.transpose(transformations[0:3,3,i]),rotation)
                transformations[0:3,3,i] = np.transpose(transl_rot)                
           
            tol_new = getG(X_p,dim,n_points,n_samples);
        
        # Scaling
        print('Scaling')
        if(scale_flag ==1) : 
            beta = np.zeros((n_samples))
            vecXp = np.reshape(X_p,(n_points*dim, n_samples)) # Columns are observations, each row is a coordinate
            Phi = np.corrcoef(vecXp, rowvar=False);
            [eigval,eigvec] = np.linalg.eig(Phi)
            sorted_id = np.argsort(eigval)
            largest_eigvec = eigvec[:,sorted_id[-1]] # eigenvector of largest eigenvalue of corr matrix
            
            sumOfSquaredNorms = 0
            for i in range(n_samples):
                sumOfSquaredNorms = sumOfSquaredNorms + math.pow(np.linalg.norm(X_p[:,:,i]),2);
            for i in range(n_samples):
                norm_sqr = math.pow(np.linalg.norm(X_p[:,:,i]),2)
                beta[i] = math.sqrt(sumOfSquaredNorms/norm_sqr)*abs(largest_eigvec[i]);
                
                # Update transformation matrix
                transformations[:,:,i] =  transformations[:,:,i]*beta[i]
                X_p[:,:,i] = beta[i] * X_p[:,:,i];
            tol_new = getG(X_p,dim,n_points,n_samples)
        
        print(abs(tol_old - tol_new))
    
    # Correct transformation matrix
    for i in range(n_samples):
        transformations[3,3,i] = 1
    X_bar = np.mean(X_p,2) # Final mean shape
    
    return transformations, X_bar


def GPA_shape(shape_list,scale,dim,threshold):
    
    n_landmarks = shape_list[0].shape[0]
    n_shapes = len(shape_list)

    # Reshape for approprite order in GPA function
    matrix_reshaped = np.zeros((n_landmarks,dim,n_shapes))
    for id_,shape in enumerate(shape_list):
        matrix_reshaped[:,:,id_] = shape
    
    transformations, mean_shape = GPA(matrix_reshaped,scale,threshold)
    
    transf_list_shape = list()
    for shape in shape_list:
        procrustes_res = procrustes(mean_shape, shape,scaling=scale,reflection=False)
        d,Z,tform = procrustes_res
        transf_shape = apply_transformations(tform,shape)
        transf_list_shape.append(transf_shape)
        full_mat_reg = mat2flat(transf_list_shape)

    return full_mat_reg, mean_shape
   
    
def bounding_box_from_center(center,width,dim):    


    # center is 3d/2d vector
    # width is 3d/2d vector with width for each axis 
    if(dim==3):
        vertices = np.zeros((8,3))
        
        vertices[0,:] = center + width
        vertices[1,:] = center + np.array([width[0],width[1],-width[2]])
        vertices[2,:] = center + np.array([width[0],-width[1],width[2]])
        vertices[3,:] = center + np.array([-width[0],width[1],width[2]])
        vertices[4,:] = center + np.array([-width[0],-width[1],width[2]])
        vertices[5,:] = center + np.array([-width[0],width[1],-width[2]])
        vertices[6,:] = center + np.array([width[0],-width[1],-width[2]])
        vertices[7,:] = center + np.array([-width[0],-width[1],-width[2]])
        
    elif(dim==2):
        # coordinate z is left as 0 to work with 3d tools
        vertices = np.zeros((4,3))
       
        vertices[0,0:2] = center + width
        vertices[1,0:2] = center + np.array([-width[0],-width[1]])
        vertices[2,0:2] = center + np.array([width[0],-width[1]])
        vertices[3,0:2] = center + np.array([-width[0],width[1]])
        
        
    return vertices
    
def procrustes(X, Y, scaling=True, reflection='best'):


       n,m = X.shape
       ny,my = Y.shape

       muX = X.mean(0)
       muY = Y.mean(0)

       X0 = X - muX
       Y0 = Y - muY

       ssX = (X0**2.).sum()
       ssY = (Y0**2.).sum()

       # centred Frobenius norm
       normX = np.sqrt(ssX)
       normY = np.sqrt(ssY)

       # scale to equal (unit) norm
       X0 /= normX
       Y0 /= normY

       if my < m:
           Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

       # optimum rotation matrix of Y
       A = np.dot(X0.T, Y0)
       U,s,Vt = np.linalg.svd(A,full_matrices=False)
       V = Vt.T
       T = np.dot(V, U.T)

       if reflection is not 'best':

           # does the current solution use a reflection?
           have_reflection = np.linalg.det(T) < 0

           # if that's not what was specified, force another reflection
           if reflection != have_reflection:
               V[:,-1] *= -1
               s[-1] *= -1
               T = np.dot(V, U.T)

       traceTA = s.sum()

       if scaling:

           # optimum scaling of Y
           b = traceTA * normX / normY

           # standarised distance between X and b*Y*T + c
           d = 1 - traceTA**2

           # transformed coords
           Z = normX*traceTA*np.dot(Y0, T) + muX

       else:
           b = 1
           d = 1 + ssY/ssX - 2 * traceTA * normY / normX
           Z = normY*np.dot(Y0, T) + muX

       # transformation matrix
       if my < m:
           T = T[:my,:]
       c = muX - b*np.dot(muY, T)

       #transformation values 
       tform = {'rotation':T, 'scale':b, 'translation':c}

       return d, Z, tform

def apply_transformations(tform,shape):
    
      # tform(as the output from procrustes function)
      #      a dict specifying the rotation, translation and scaling that
      #      maps X --> Y
      R = tform['rotation']
      s = tform['scale']
      t = tform['translation']
      
      trans_shape = s*np.matmul(shape,R)+t.reshape(-1,1).T
      return trans_shape
