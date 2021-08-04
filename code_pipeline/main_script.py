# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:57:48 2021

@author: filipavaldeira
"""


if __name__ == "__main__":
        
    from pipeline import pipeline
    
    # --------------- Settings --------------- #
    # ----- Data
    path_data = r'Data\Original_data' # Path for folder with original data obj files
    path_template = r'Data\\PCA_data\1.ply' # path for template mesh file
    path_pca_data = r'Data\PCA_data'
    dim=3 # dimension of data
    id_list = 'all' # list of ids to register or 'all' to register all shapes
    
    # ----- SDRSAC Registration
    max_dist_sdrsac = 2 # max distance for inlier filtering
    max_dist_inlier_retrieval = 4 # max distance for inlier retrieval
    sdrsac_max_iter = 5 # maximum number of iterations SDRSAC
    sdrsac_print_details = False
    
    # ----- BCPD Registration
    bcpd_omega = 0.1
    bcpd_lmbd = 400
    bcpd_beta = 0.8
    bcpd_max_dist = 0.2 #Maximum radius to search for neighbors
    
    # ----- GP Registration
    gp_var_obs = 40
    gp_max_dist_search = 3
    # SE kernel paramterers
    kernel_se_l = 5
    kernel_se_var = 10    
    
    pipeline(path_data, path_template, path_pca_data, dim)
