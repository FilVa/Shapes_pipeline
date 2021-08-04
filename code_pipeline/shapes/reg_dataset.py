# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:59:47 2019

@author: filipavaldeira
"""

import numpy as np

from utils.convert import flat2mat,convert_to_dict,mat2flat
from utils.convert import correspondence_switch,array_filter_id_nan
from utils.transformations import procrustes
from utils.io import write_mesh

from scipy.spatial import KDTree
import os
import copy
import open3d as o3d
# Class for registered dataset
# each shape has the same number of points
class RegDataset(object):
    
    def __init__(self,og_dataset, template, def_src_mat,corr_vec_list, og_data, id_vec, reg_method,reg_time,parameters_str):
        
        # Data matrices
        self.template = template
        self.original_data = og_data
        self.corr_vec_list = corr_vec_list
        self.og_dataset = og_dataset
        
                
        self.dim = template.shape[1]
        self.n_points = template.shape[0]
        self.n_obs = og_dataset.n_samples
        self.reg_time = reg_time
        self.parameters_str = parameters_str
        
        self.id_vec = np.asarray(id_vec).astype('int')
        self.reg_method = reg_method

        self.corr_by_template = convert_to_dict(self.id_vec,corr_vec_list)
        self.corr_by_target = self.get_corr_by_target()

        self.def_src_dict = self.flat_to_mat_dic(convert_to_dict(self.id_vec,def_src_mat))
        
        self.og_template_non_ass, self.def_template_non_ass = self.get_templates_not_assigned()
        self.target_non_ass = self.get_target_not_assigned()
        
        self.landmark_matrix,self.og_template_ass,self.def_template_ass = self.get_landmark_matrix() # samples with registered points
        
    def flat_to_mat_dic(self,dic):
        for id_ in dic.keys():
            dic[id_] = flat2mat(dic[id_].reshape((1,-1)),self.n_points,self.dim)
        return dic
        
        
    def get_landmark_matrix(self):        
        landmark_list = list()        
        og_template_ass = dict()
        def_template_ass = dict()
        for id_, shape in self.og_dataset.shapes_dict.items():
            corr = self.corr_by_template[id_]
            reg_pts = array_filter_id_nan(shape.points,corr)
            landmark_list.append(reg_pts)
            
            template_pts = copy.copy(self.template)
            template_pts[np.isnan(corr),:]=np.nan
            og_template_ass[id_] = template_pts
            
            def_template_pts =  copy.copy(self.def_src_dict[id_])
            def_template_pts[np.isnan(corr),:]=np.nan
            def_template_ass[id_] = def_template_pts           
            
        landmark_mat = mat2flat(landmark_list)
        return landmark_mat,og_template_ass,def_template_ass
        
    def get_corr_by_target(self):
        new_corr = dict()
        for id_ in self.corr_by_template.keys():
            new_corr[id_] = correspondence_switch(self.corr_by_template[id_],self.og_dataset.shapes_dict[id_].get_n_points())
        return new_corr   

    def get_templates_not_assigned(self):
        og_template = dict()
        def_template = dict()
        for id_,corr in self.corr_by_template.items():
            mask_nan = np.isnan(corr)       
            og_template[id_] = self.template[mask_nan,:]
            def_src = self.def_src_dict[id_]
            def_template[id_] = def_src[mask_nan,:]            
        return og_template,def_template        
    
    def get_target_not_assigned(self):
        target_dic = dict()
        for id_,corr in self.corr_by_target.items():
            mask_nan = np.isnan(corr)       
            target_dic[id_] = self.og_dataset.shapes_dict[id_].points[mask_nan,:]
        return target_dic      
        

    def get_og_dataset_reg_points_only(self):
        og_dataset = self.og_dataset
        corr_vec = self.corr_vec_list
        
        for id_ in og_dataset.shapes_dict.keys():
            idx = np.where(self.id_vec == id_)
            corr = corr_vec[idx[0][0]]
            unique_ids = np.unique(corr[~np.isnan(corr)]).astype('int')
            
            
            og_dataset.shapes_dict[id_].points = og_dataset.shapes_dict[id_].points[unique_ids,:]
            og_dataset.corresp_dict[id_] = og_dataset.corresp_dict[id_][unique_ids]
            
        return og_dataset
    
    def save_results(self,dest_folder,dest_template_pos,file_type,max_dist,scale=False):
        og_dataset = copy.copy(self.og_dataset)

        for id_, shape in og_dataset.shapes_dict.items():
            print('Saving shape : {} '.format(id_))
        
            def_template = self.def_src_dict[id_]
            procrustes_res = procrustes(dest_template_pos, def_template,scaling=scale,reflection=False)
            d,Z,tform = procrustes_res            
            shape.apply_rigid_transform(tform)        

            trans_mesh = shape.get_o3d_mesh()
            mesh_path = os.path.join(dest_folder,'Meshes', str(id_)+file_type)
            o3d.io.write_triangle_mesh(mesh_path, trans_mesh,print_progress =True)
            
            tree = KDTree(shape.points)
            dist,ids = tree.query(dest_template_pos,k=1,distance_upper_bound=max_dist)
            final_pts = shape.points[ids[dist!=np.inf],:]
            file_path = os.path.join(dest_folder, 'Inliers' , str(id_)+'.csv')
            np.savetxt(file_path, final_pts,delimiter=',')            

    
    def save_shapes_to_files_transformed(self,dest_folder,file_type,flag_as_mesh,dest_template_pos):
        # flag_as_mesh = if true save as mesh, otherwise save as point cloud (must be coherent with file type)
        print("ONLY WORKS FOR MESHSES RIGTH NOW")
        
        og_dataset = copy.copy(self.og_dataset)
        
        for id_, shape in og_dataset.shapes_dict.items():
            print(id_)
            def_template = self.def_src_dict[id_]
            procrustes_res = procrustes(dest_template_pos, def_template,scaling=False,reflection=False)
            d,Z,tform = procrustes_res            
            shape.apply_rigid_transform(tform)
            trans_mesh = shape.get_o3d_mesh()
            mesh_path = os.path.join(dest_folder, str(id_)+file_type)
            o3d.io.write_triangle_mesh(mesh_path, trans_mesh,print_progress =True)

    def save_shapes_to_files_transformed_only_reg_pts(self,dest_folder,dest_template_pos,max_dist,scale=False):
        
        og_dataset = copy.copy(self.og_dataset)        
        
        for id_, shape in og_dataset.shapes_dict.items():
            print(id_)
            def_template = self.def_src_dict[id_]
            procrustes_res = procrustes(dest_template_pos, def_template,scaling=scale,reflection=False)
            d,Z,tform = procrustes_res            
            shape.apply_rigid_transform(tform)
            
            tree = KDTree(shape.points)
            dist,ids = tree.query(dest_template_pos,k=1,distance_upper_bound=max_dist)
            final_pts = shape.points[ids[dist!=np.inf],:]
            file_path = os.path.join(dest_folder, str(id_)+'.csv')
            np.savetxt(file_path, final_pts,delimiter=',')
            
    def get_inliers(self,dest_folder,max_dist):
        og_dataset = copy.copy(self.og_dataset)
        for id_, shape in og_dataset.shapes_dict.items():
            print(id_)
            def_template = self.def_src_dict[id_]
            tree = KDTree(def_template)
            dist,ids = tree.query(shape.points, k=1,distance_upper_bound=max_dist)
            final_pts = shape.points[dist!=np.inf,:]
            file_path = os.path.join(dest_folder, str(id_)+'.csv')
            np.savetxt(file_path, final_pts,delimiter=',')
            
       
    def save_deformed_template(self,dest_folder,faces):
        for id_, shape in self.def_src_dict.items():
            file_path = os.path.join(dest_folder, str(id_)+'.ply')
            write_mesh(shape,faces,file_path)
    