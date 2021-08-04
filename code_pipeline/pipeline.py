# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:57:48 2021

@author: filipavaldeira
"""


import os
from registration.sdrsac_python import SdrsacPyRegistration
from registration.bcpd import BcpdRegistration    
from utils.io import read_data_folder_obj, read_data_folder_csv, read_mesh, setup_directories
from gp_framework.gp_kernels import EmpiricalKernel, TotalKernel
from gp_framework.gp_general import gp_shape_completion_dataset
import gpflow
    
    
def pipeline(path_data, path_template, path_pca_data, dim, id_list = 'all', 
             max_dist_sdrsac = 2, max_dist_inlier_retrieval = 4, sdrsac_max_iter = 5, 
             bcpd_omega = 0.1, bcpd_lmbd = 400, bcpd_beta = 0.8, bcpd_max_dist = 0.2, 
             gp_var_obs = 40, gp_max_dist_search = 3, kernel_se_l = 5, kernel_se_var = 10,
             sdrsac_print_details = False):

    
    print('----------------------------------------------------------------')
    print(' Step 0 : Setup directories and preparing data ')
    print('----------------------------------------------------------------')

    # Get main directory path
    main_dir = os.getcwd()
    path_sdrsac_results, path_sdrsac_meshes, path_sdrsac_inliers, path_bcpd_results, path_gp_results = setup_directories(main_dir)
        
    # --------------- Read data --------------- #  
    dataset_og = read_data_folder_obj(path_data, '',dim)
    template,template_faces = read_mesh(path_template,remove_degenrate_flag=False)        
    
    # ------ Step 1 registration with SDRSAC ----- #
    print('----------------------------------------------------------------')
    print(' Step 1 : Performing SDRSAC Registration ')
    print('----------------------------------------------------------------')
    print('>>>>>>> Resgistering with SDRSAC')
    sdrsac_reg = SdrsacPyRegistration(reg_type = 'point_to_point',max_dist=max_dist_sdrsac, MAXITER = sdrsac_max_iter,scaling = True, print_details = sdrsac_print_details)
    reg_out_step1 = sdrsac_reg.simple_registration(template, dataset_og, id_list, flag_parallel=False)    
    print('>>>>>>> Saving SDRSAC Results')
    reg_out_step1.save_results(path_sdrsac_results,template,'.ply', max_dist_inlier_retrieval)
    
    # ------ Step 2 registration with BCPD ----- #
    print('----------------------------------------------------------------')
    print(' Step 1 : Performing BCPD Registration ')
    print('----------------------------------------------------------------')    
    print('>>>>>>> Reading BCPD data')
    dataset = read_data_folder_csv(path_sdrsac_inliers, '', dim)
    bcpd_reg = BcpdRegistration(omega = bcpd_omega, lmbd = bcpd_lmbd, beta = bcpd_beta, dist = bcpd_max_dist, flag_std_acc = 1)
    print('>>>>>>> Resgistering with BCPD')
    reg_out_step2 = bcpd_reg.simple_registration(template, dataset, id_list='all', flag_parallel = False)
    print('>>>>>>> Saving BCPD Results')
    reg_out_step2.save_deformed_template(path_bcpd_results,template_faces)
    
    # ------ Step 3 Shape completion with GP ----- #
    print('----------------------------------------------------------------')
    print(' Step 3 : Shape completion with GP ')
    print('----------------------------------------------------------------')    
    # Read PCA data
    print('>>>>>>> Reading PCA data')
    pca_ids, pca_shape_list = read_data_folder_obj(path_pca_data, '',dim).get_shape_pts_list('all')
    # kernel setting
    k_pca =  EmpiricalKernel(template,pca_shape_list, dim, n_components=3)
    k_exp =  gpflow.kernels.SquaredExponential(variance = kernel_se_var,lengthscales = kernel_se_l)
    k = k_pca + k_exp
    full_kernel = TotalKernel(kernel_list=k,dim=dim)    
    # run shape completion
    print('>>>>>>> Shape completing')
    gp_shape_completion_dataset(reg_out_step2, full_kernel, path_template, path_sdrsac_inliers, path_gp_results, gp_max_dist_search, gp_var_obs)
    
    print('----------------------------------------------------------------')
    print(' Pipeline completed ')
    print('----------------------------------------------------------------')      
     
    
