# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:46:12 2020

@author: filipavaldeira
"""


import numpy as np
import pandas as pd
import os
from utils.convert import correspondence_switch
import matlab.engine

from registration.registration import ShapesRegistration

class BcpdRegistration(ShapesRegistration): 
    def __init__(self,omega = 0.1, lmbd=1e4,beta =2.0,gamma = 3.0,cov_tol = 1e-4,min_VB_loops = 30,max_VB_loops = 500,dist = 0.15,flag_std_acc=0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.reg_method = 'BCPD'
        
        # BCPD Paramterers with default values from original code
        self.omega = omega # outlier probability
        self.lmbd = lmbd # expected length of deformation
        self.beta = beta # parameter of gaussian kernel
        self.gamma = gamma # randomness of pt matching (positive)
        self.dist=dist # Maximum radius to search for neighbors.
        
        # convergence BCPD parameters
        self.cov_tol = cov_tol
        self.max_VB_lopps = max_VB_loops
        self.min_VB_loops = min_VB_loops
        
        # acceleration
        self.acc = flag_std_acc # activates acceleration with standard parameters
            
        self.parameters_str = 'Omega = {}. Lambda = {}'.format(self.omega,self.lmbd)

    
    def pair_registration(self,target,source):
        # how to perform registration between a target and a source        
        
        # Runs matlab code BCPD from original authors
        opt = dict()
        # BCPD parameters
        # Matlab needs these to be floats
        opt['omg']= float(self.omega)
        opt['lmd'] = float(self.lmbd)
        opt['beta'] = float(self.beta)
        opt['gamma'] = float(self.gamma)
        opt['dist'] = float(self.dist)
        # Convergence param
        opt['tol']= float(self.cov_tol)
        opt['max_loops'] = float(self.max_VB_lopps)
        opt['min_loops'] = float(self.min_VB_loops)
        # acceleration
        opt['flag_acc']= float(self.acc)
        

        target_path = os.path.join(os.getcwd(),r'bcpd\shape_target.txt')
        np.savetxt(target_path, target,delimiter=',')
        source_path = os.path.join(os.getcwd(),r'bcpd\shape_source.txt')
        np.savetxt(source_path, source,delimiter=',')
            
        input_dict = {'X' : target_path, 'Y' : source_path, 'opt' : opt}
        output_dict = matlab_cpd_reg(input_dict)
        target_corr = output_dict['correp_vec']
        deformed_source = output_dict['deformed_source']

        # Corr_vec returned for this method is the target one, we need to convert
        target_corr = np.reshape(target_corr,(1,-1))
        target_corr = target_corr[0]
        corr_vec = correspondence_switch(target_corr, source.shape[0])
        deformed_source =  np.reshape(deformed_source, (1,-1))
        
        return deformed_source, corr_vec
    
    def read_template(self,template_path):
        df = pd.read_csv(template_path,header=None )
        template = df.to_numpy()
        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        self.template = template
   

# Calls matlab function for CPD registration
def matlab_cpd_reg(input_dict):
    output_dict = dict()
    
    X = input_dict['X']
    Y = input_dict['Y']
    opt = input_dict['opt']

    eng = matlab.engine.start_matlab()
    
    bcpd_loc = os.path.join(os.getcwd(),r'bcpd')
    eng.addpath(eng.genpath(bcpd_loc))  
    eng.cd(bcpd_loc)

    transform_Y, correspondence_vec = eng.bcpd_register(X,Y,opt, nargout=2)
    
    eng.exit()
    correspondence_vec = np.asarray(correspondence_vec)-1

    output_dict['deformed_source'] = np.asarray(transform_Y)
    output_dict['correp_vec'] = correspondence_vec
    
    return output_dict


    
    
    
    