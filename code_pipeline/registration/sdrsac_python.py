# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:31:43 2019

@author: filipavaldeira
"""

import copy
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import special_ortho_group
import open3d as o3d 

# Local imports
from registration.registration import ShapesRegistration
from utils.convert import numpy2PointCloud

### Auxiliar functions
def aux_correspondence(reg_p2p,sz):
    correspondence_vec = np.array(reg_p2p.correspondence_set)
    keep_id = correspondence_vec[:,1]
    corr_vec = np.empty((sz))
    corr_vec[:]  = np.nan
    corr_vec[correspondence_vec[:,0]] = keep_id
    return corr_vec

###############################################################################        
## ---------------------------------- SdrsacPyRegistration ---------------------------------- ##         
############################################################################### 
        
class SdrsacPyRegistration(ShapesRegistration):
    def __init__(self,  reg_type, max_dist, initial_transf = None, scaling =False, MAXITER = 500, print_details =False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.reg_type = reg_type 
        self.threshold = max_dist
        self.print_details = print_details
        
        if(self.reg_type == 'point_to_point'):
            self.reg_scale = scaling
        elif(self.reg_type == 'point_to_plane'):
            self.reg_scale = None        
        
        self.parameters_str = 'Max dist = {}. Scaling = {}'.format(max_dist,scaling)
        self.MAX_ITER = MAXITER
        
    def pair_registration(self,target_og,source_og):        
        
        # Estimate normals for point clouds
        src, target = numpy2PointCloud(source_og), numpy2PointCloud(target_og)
        src.estimate_normals()
        target.estimate_normals()
        normals_target = np.asarray(target.normals)
        
        # Translation between center of mass of point clouds
        cm_src = source_og.mean(axis=0)
        cm_target = target_og.mean(axis=0)
        t_cm =cm_target- cm_src 
        
        # set initial values        
        template_shape = np.array(src.points).shape
        tree = KDTree(target_og)
        stop = False # flag to leave iterations
        i = 0 # iteration counter
        tmax = 10e9 # initialize maximum iterations estimate with high value
        best_inliers = 0 # save number of inliers for best iteration
        ps = 0.99
        eps = self.threshold 
        dim = source_og.shape[1]
        icp_init_transf = np.eye(dim+1)
        
        while(stop == False):                                           
            #Random rotation of template
            if(i>0):
                R = special_ortho_group.rvs(self.dim)
                T_source = np.matmul(R,(source_og-cm_src).T).T+cm_src + t_cm
            else:
                # first iteration with initial position (in case it is already close enough)
                T_source = source_og
            
            # compute normals of transformed template
            src = numpy2PointCloud(T_source)
            src.estimate_normals()
            
            # ICP registration different types
            if(self.reg_type == 'point_to_point'):
                reg_p2p = o3d.pipelines.registration.registration_icp(
            src, target, self.threshold, icp_init_transf,o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=self.reg_scale))            
            elif(self.reg_type == 'point_to_plane'):
                reg_p2p = o3d.pipelines.registration.registration_icp(
            src, target, self.threshold, icp_init_transf,o3d.pipelines.registration.TransformationEstimationPointToPlane())                        
            
            #Correspondence vector handling
            corr_vec = aux_correspondence(reg_p2p,template_shape[0])
    
            # obtain deformed source and normals
            transf_source = copy.deepcopy(src)
            transf_source = transf_source.transform(reg_p2p.transformation)
            transf_source.estimate_normals()
            normals_template = np.asarray(transf_source.normals)
            def_src = np.asarray(transf_source.points)            
            
            # Get correspondences within distance
            dd,ii = tree.query(def_src[:,0:self.dim],k=1,distance_upper_bound=eps)
            mask_inliers = ~np.isinf(dd)
            
            # Get number of accepted inliers
            good_inliers = self.accepted_inliers(normals_template,normals_target,mask_inliers,ii)

            if(good_inliers>best_inliers):
                if self.print_details : print('-> Found new best tranformation') 
                # Keep results                
                best_inliers = good_inliers
                best_corr_vec = corr_vec
                best_deformed_source = np.reshape(def_src[:,0:self.dim],(1,-1)) #remove stub zeros and reshape
                
                # Update expected iterations
                pi = good_inliers/source_og.shape[0]
                if(pi==1):
                    tmax=1 
                else:
                    tmax = np.log(1-ps)/np.log(1-pi**4)
                    
                if self.print_details : print('Number of inliers: {}'.format(best_inliers))
                if self.print_details : print('Updated expected number of iterations : {}'.format(tmax))
                
            # If reached number of maximum iterations allowed or expected iterations leave cycle
            if((i>tmax)|(i>self.MAX_ITER)):
                stop=True            
            i += 1        
        return best_deformed_source,best_corr_vec
    
    def accepted_inliers(self,normals_template,normals_target,mask_inliers,ii):
        in_normals_template = normals_template[mask_inliers,:]
        in_normals_target = normals_target[ii[mask_inliers],:]
        # compute angle between normals
        inner_p = np.sum(in_normals_template*in_normals_target,axis=1)
        ang_matrix_rad = np.arctan2( np.linalg.norm(np.cross(in_normals_template,in_normals_target),axis=1),inner_p)
        sin_mat = np.sin(ang_matrix_rad)
        good_inliers =np.sum( np.abs(sin_mat)<np.sin(np.pi/4))
        return good_inliers
