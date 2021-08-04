# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:46:04 2019

@author: filipavaldeira
"""

from shapes.sample import Shape
import open3d as o3d
import numpy as np
import os
from utils.convert import array_filter_id_nan

# Cretaes new dataset from dictionary of points and dictionary of correspondences
def create_dataset(new_pts_dict,new_corr_dict,dim):
    
    new_dataset = ShapeDataset(dim)
    for id_, pts in new_pts_dict.items():
        new_dataset.add_shape(int(id_), pts)
        corr = new_corr_dict[id_]
        new_dataset.add_corresp(int(id_),corr)      
    
    return new_dataset
        
        
#################### CLASS Dataset ####################
class ShapeDataset(object):
    
    def __init__(self, dim):       
        self.dim = dim
        self.n_samples = 0
        self.shapes_dict = dict() # list of shapes objects
        self.mesh_dict = dict()
       
        # For each shape is an array with same number of points of the shape. If shape has outliers NaN, if missing points then the id is not present in the array
        self.corresp_dict = dict() 
        self.dataset_type = 'non_def' # point cloud or mesh, here is still not defined

    def add_mesh(self,id_,vertices,triangles):
        if(self.dataset_type=='point_cloud'):
            print('Do not mix dataset of pointcloud and mesh. will not add new shape')
        else :
            self.dataset_type='mesh'
            print('Adding new shape to dataset: Mesh')
            shape = Shape()
            shape.add_points(vertices)
            shape.add_triangles(triangles)
            self.shapes_dict[id_] = shape
            self.n_samples = self.n_samples + 1       
            # Init correspondence with ids
            self.add_corresp(id_,np.arange(vertices.shape[0]))    
    
    def add_shape(self,id_,points): 
        if(self.dataset_type=='mesh'):
            print('Do not mix dataset of pointcloud and mesh. will not add new shape')
        else :
            self.dataset_type='point_cloud'

            print('Adding new shape to dataset: Point Cloud')
            shape = Shape()
            shape.add_points(points)
            self.shapes_dict[id_] = shape
            self.n_samples = self.n_samples + 1 
            
            # Init correspondence with ids
            self.add_corresp(id_,np.arange(points.shape[0]))
            
    def add_corresp(self,id_,points):
        self.corresp_dict[id_] = points
    
    def add_list_shapes(self,id_list,pts_list):
        for id_,shape in zip(id_list,pts_list):
            self.add_shape(id_,shape)   
    
    def update_corr_after_template_change(self, new_template_ids):
        for id_, corr_vec in self.corresp_dict.items():
            new_corr = array_filter_id_nan(new_template_ids,corr_vec)
            self.corresp_dict[id_] = new_corr

    # Returns list with id of all shapes        
    def get_id_list(self):
        return list(self.shapes_dict.keys())

    # Return shapes in id_list
    def get_shape_pts_list(self,id_list):
        if(isinstance(id_list,list)):
            shape_list = [self.shapes_dict[x].points for x in id_list]
            ids = id_list
        elif(id_list=='all'):
            ids = list(self.shapes_dict.keys())
            shape_list = [self.shapes_dict[x].points for x in ids]
        elif(isinstance(id_list,int)):
            ids = id_list
            shape_list = [self.shapes_dict[ids].points]
            
        return ids, shape_list
    
    # Returns list with points of id_points for every shape
    def get_shape_points_by_id(self,id_points):
        shape_list = [shape.points[id_points,:] for shape in self.shapes_dict.values()]
        return shape_list  
    
    def set_origin_by_user_pt(self,point):
        for id_, shape in self.shapes_dict.items():
            shape.set_origin_by_user_pt(point)
    
    def apply_rigid_transform(self,tform):
        for id_, shape in self.shapes_dict.items():
            shape.apply_rigid_transform(tform)
