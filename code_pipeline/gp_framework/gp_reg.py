# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:46:36 2021

@author: filipavaldeira
"""
import numpy as np
import gpflow
import time
import open3d as o3d
import os

from scipy.spatial import KDTree

from utils.convert import mat2flat
from utils.io import read_mesh


def find_closest_pts(ref,target,max_dist):
    tree = KDTree(target)
    neighbor_dists, neighbor_indices = tree.query(ref,distance_upper_bound=max_dist)
    return neighbor_dists, neighbor_indices


def get_standard_observations(ref_shape,target_shape,max_dist,original_ref_shape):
    
    dist,ids = find_closest_pts(ref_shape,target_shape, max_dist)
    
    filtered_dist = dist[~np.isinf(dist)]
    filtered_ids_target = ids[~np.isinf(dist)]
    
    ref_shape_observations = original_ref_shape[~np.isinf(dist),:]
    target_shape_observation = target_shape[filtered_ids_target]
    deformations = target_shape_observation-ref_shape_observations
    
    return ref_shape_observations, deformations



class GpModel(object):
    def __init__(self,ref_shape_path,kernel,mean_function=None,flag_keep_for_test = True):
        if(isinstance(ref_shape_path, str)):
            self.ref_shape,self.ref_shape_faces = read_mesh(ref_shape_path,remove_degenrate_flag=False)
            self.ref_shape_path = ref_shape_path
        else:
            self.ref_shape=ref_shape_path
            
        self.kernel = kernel
        self.mean_function= mean_function
        self.flag_keep_for_test = flag_keep_for_test


    def iterative_reg(self,target_shape,max_dist_search,var_obs,maxiter,flag_first_iter_def=False,first_iter_pts = None,first_iter_def=None):
        
        MAX_ITER = maxiter
        max_dist = max_dist_search
        K = self.kernel
        var_observations = var_obs
        error_tolerance = 0.00001      
        
        M = self.ref_shape.shape[0]
        dim = self.ref_shape.shape[1]
        
        ref_shape_og = self.ref_shape.copy()

        # Initial settings for cycle
        flag_leave = False
        n_iter = 0
        current_ref_shape = ref_shape_og.copy()
        ref_shape = ref_shape_og.copy()
        ref_shape_flat = mat2flat(ref_shape_og)                        
        
        # ----- List variables to save results
        self.shapes_list = list()
        self.mean_err_to_prev, self.max_err_to_prev = list(), list() # Store error with respect to previous shape,
        self.iter_time, self.n_def, self.current_likelihood = list(), list(), list()
        self.var_obs = var_obs        
                
        while(flag_leave==False):
            start = time.time()
            
            #STEP 1 : Get poitns and respective deformations for GP regression
            if((flag_first_iter_def == True)&(n_iter==0)):
                points = first_iter_pts
                deformations = first_iter_def
            else:
                points,deformations = get_standard_observations(current_ref_shape,target_shape,max_dist,ref_shape)
            
            n_def = points.shape[0]    
            if(n_def==0):
                return self.shapes_list.copy(),target_shape,ref_shape
            
            # STEP 2 :  Compute GP regression
            X = points.copy() # points where we put the deformations
            Y = deformations.copy() # observations = deformations        
            X_flat = mat2flat(X)
            Y_flat = mat2flat(Y)        

            model = gpflow.models.GPR(data=(X_flat.T, Y_flat.T), kernel=K, mean_function=self.mean_function,noise_variance= var_observations)
                          
            mean, var = model.predict_f(ref_shape_flat.T)      
            mean = mean.numpy()
            deformed_template = ref_shape + np.reshape(mean,(-1,dim))

            # stop criterion
            errors = np.linalg.norm(deformed_template-current_ref_shape,axis=1)
            delta = errors.mean()
            n_iter += 1
            
            if((errors.mean()<error_tolerance)|(n_iter==MAX_ITER)):
                flag_leave = True
            
            current_ref_shape = deformed_template.copy()
            end = time.time()
            
            self.shapes_list.append(current_ref_shape)
            self.mean_err_to_prev.append(delta)
            self.max_err_to_prev.append(errors.max())
            self.iter_time.append(end-start)
            self.n_def.append(n_def)

        return self.shapes_list.copy(),target_shape,ref_shape
    
    def write_obj_results(self,folder,general_name):
        
        self.output_folder = folder
        self.output_files = list()
                
        for id_,shape in enumerate(self.shapes_list):
            mesh  =o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(shape),triangles =o3d.utility.Vector3iVector(self.ref_shape_faces))
            mesh.compute_vertex_normals()

            file_name = general_name+str(id_)+'.ply'
            path = os.path.join(folder, file_name)
            self.output_files.append(file_name)
            o3d.io.write_triangle_mesh(path, mesh)     
        
            
            
            
            
            
            


        
        