# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:31:33 2019

@author: filipavaldeira
"""

import copy
import numpy as np
import time
from multiprocessing import Pool # Multiprocessing

from utils.io import read_mesh
from shapes.reg_dataset import RegDataset # REPLACE RegOutpu

###############################################################################
##################### General Class for Shape Registration ####################
###############################################################################


class ShapesRegistration(object):
    def __init__(self):
        self.id_vec = None # id in the same order as points are found on data matrices
        self.original_data = None # original data points as read from the files (list of arrays, each with possibly different number of points)
        
        self.dim = None
        self.n_points = None
        
        # Data matrices
        self.template = None
        self.landmark_matrix = None # samples with registered points
        self.deformed_source_matrix = None
        self.target_non_assigned_pts = None              
        self.reg_method = ''

    # Register all files in src_path against template path
    def simple_registration(self,template, dataset, id_list,flag_parallel):
        id_vec, data = dataset.get_shape_pts_list(id_list)        
        
        self.original_data = data
        self.id_vec = id_vec
        self.template = template
        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        # Register matrix
        deformed_src_matrix,corr_vec_list,reg_time = self.register_matrix(self.template,self.original_data,self.id_vec,flag_parallel)

        self.deformed_source_matrix = deformed_src_matrix

        reg_output = RegDataset(dataset,self.template, deformed_src_matrix, corr_vec_list, data, id_vec, self.reg_method,reg_time,self.parameters_str)
        
        return reg_output      
    
    def register_matrix(self,template,data,id_vec, flag_parallel):

        deformed_src_matrix = None
        corr_vec_list = list()
        data_list = list()
        output = list()
        num_processors = 3
        
        # create input list
        for n_sample in range(len(id_vec)):
            element_list = list()
            if(isinstance(data, list)):
                source = data[n_sample]
            else:
                source = data[:,:,n_sample] 
            element_list.append(source)
            element_list.append(template)
            element_list.append(id_vec[n_sample])
            data_list.append(element_list)
            
        start = time.time()
        if(flag_parallel == True):
            # call registration in parallel
            print('Starting {} parallel processes'.format(num_processors))
            p = Pool(processes = num_processors)
            output = p.map(self.reg_process,[i for i in data_list])            
        else:
            print('Start registration')
            # Registration in series
            for data_input in data_list:
                print('---- Registering shape with ID : {}'.format(data_input[-1]))
                out = self.reg_process(data_input)
                output.append(out)
        end = time.time()
        
        print('End all registration {}'.format(end-start))
        reg_time = end-start
        
        # Handle output list
        for element in output:
            if element is None:
                print('Not matched, wrong results') 
                vec_zeros = np.zeros((1,template.shape[0]*template.shape[1]))
            # deformed matri
            else:

                deformed_src = element[0]
                corr_vec = element[1]
                # Save id and vertex matrix
                if deformed_src_matrix is None :
                    deformed_src_matrix = copy.copy(deformed_src)
                    corr_vec_list.append(corr_vec)
                else:

                    deformed_src_matrix = np.append(deformed_src_matrix,deformed_src,axis=0)
                    corr_vec_list.append(corr_vec)

        return deformed_src_matrix,corr_vec_list,reg_time
    
    # PRocess for registration
    def reg_process(self,input_list):
        source = input_list[0]
        template = input_list[1]
        id_ = input_list[2]
        output_list = list()
        
        start = time.time()
        #print ("Starting registration with {} of subject: {} ".format(self.reg_method,id_))        
        deformed_src,corr_vec = self.pair_registration( source, template)
        if(deformed_src is None):
            return None
        end = time.time()
        print ("Registration time: {:.4f} ".format(end-start)) 

        output_list.append(deformed_src)
        output_list.append(corr_vec)
                
        return output_list
    
  