# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:11:18 2021

@author: filipavaldeira
"""
import numpy as np
import pandas as pd
import os

from gp_framework.gp_reg import GpModel
from utils.io import read_mesh

def gp_shape_completion_dataset(reg_out_bcpd,real_full_kernel,template_path,inliers_path,results_path,max_dist_search,var_obs):

    template_pts,template_faces = read_mesh(template_path,remove_degenrate_flag=False)
    dim = template_pts.shape[1]
    
    for id_ in reg_out_bcpd.id_vec:
        
        corr_vec = reg_out_bcpd.corr_by_template[id_]
        n_corr = np.sum([~np.isnan(corr_vec)])
        
        if(n_corr==0):
            print('No correspondence obtained for id_ : {} -> will skip shape'.format(id_))
        else:
            print('Shape completion for id_ : {}'.format(id_))
            gp_model = GpModel(template_path,real_full_kernel,None)
            
            ### Get correspondences
            inliers_path_shape = os.path.join(inliers_path,str(id_)+'.csv')
            df = pd.read_csv(inliers_path_shape,header=None )
            inliers_pts = df.values[:,0:dim]     
            
            corr_vec = reg_out_bcpd.corr_by_template[id_]
            corr_target_id= corr_vec[~np.isnan(corr_vec)].astype(int)
            corr_template_id = np.transpose(np.argwhere(~np.isnan(corr_vec)))[0].astype(int)
            
            first_iter_pts = template_pts[corr_template_id,:]
            first_iter_def =  inliers_pts[corr_target_id,:] - template_pts[corr_template_id,:]
        
            ### Read target and landmarks file
            target_points = inliers_pts
            
            def_templates_list = gp_model.iterative_reg(target_points,max_dist_search,var_obs,1,flag_first_iter_def=True,first_iter_pts = first_iter_pts,first_iter_def=first_iter_def)
            
            # Save results
            gp_model.write_obj_results(results_path, str(id_))
            
