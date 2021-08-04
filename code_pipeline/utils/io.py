# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:38:22 2019

@author: filipavaldeira
"""

import numpy as np
import os
import pandas as pd
import open3d as o3d
import shutil

from utils.convert import check_matrix
from shapes.dataset import ShapeDataset

def setup_directories(main_dir):
    path_sdrsac_results = os.path.join(main_dir,r'Data\Step_1_results')
    path_sdrsac_meshes, path_sdrsac_inliers = os.path.join(path_sdrsac_results,r'Meshes'), os.path.join(path_sdrsac_results,r'Inliers')
    path_bcpd_results = os.path.join(main_dir,r'Data\Step_2_results')
    path_gp_results = os.path.join(main_dir,r'Data\Step_3_results')
    
    # Remove existing results and directories
    shutil.rmtree(path_sdrsac_results)
    shutil.rmtree(path_bcpd_results)
    shutil.rmtree(path_gp_results)
    
    # Setup required directories
    os.mkdir(path_sdrsac_results)
    os.mkdir(path_sdrsac_inliers)
    os.mkdir(path_sdrsac_meshes)
    os.mkdir(path_bcpd_results)
    os.mkdir(path_gp_results)
    
    return path_sdrsac_results, path_sdrsac_meshes, path_sdrsac_inliers, path_bcpd_results, path_gp_results


def write_mesh(pts,faces,path):    
    mesh  =o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(pts),triangles =o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)

def pts_to_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [ avg_dist,avg_dist*1.5]
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector(radii) )

    return mesh    

def read_mesh(path,remove_degenrate_flag):
    
    mesh = o3d.io.read_triangle_mesh(path,print_progress =True)
    if(remove_degenrate_flag==True):
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    return vertices,faces
    


def read_data_folder(src_path, exclude_path,dim):
    print("OUTDATED REPLACE WITH read_data_folder_obj")
    dataset = ShapeDataset(dim)
    
    for file in os.listdir(src_path):
        mesh_file = os.path.join(src_path, file)
        if file.endswith('.ply'):
            mesh = o3d.io.read_triangle_mesh(mesh_file,print_progress =True)
        elif file.endswith('.obj'):   
            mesh = o3d.io.read_triangle_mesh(mesh_file,print_progress =True)
        elif file.endswith('.stl'):   
            mesh = o3d.io.read_triangle_mesh(mesh_file,print_progress =True)    
        else:
            continue
    
        sep = file.split('.')
        id_subj = sep[0]
        print ("Reading file of subject: {} ".format(id_subj)) 
        
        dataset.add_mesh(id_subj,mesh)
    return dataset

# Reads folder with '.ply','.stl','.obj' files and returns dataset object
def read_data_folder_obj(src_path, exclude_path,dim,flag_clean_shape=None,other_sep=None):
    dataset = ShapeDataset(dim)
    
    for file in os.listdir(src_path):
        mesh_file = os.path.join(src_path, file)
        if file.endswith('.ply'):
             mesh = o3d.io.read_triangle_mesh(mesh_file)   
        elif file.endswith('.stl'):
            mesh = o3d.io.read_triangle_mesh(mesh_file) 
        elif file.endswith('.obj'):   
            mesh = o3d.io.read_triangle_mesh(mesh_file,print_progress =True) 
        else:
            continue    
        if(flag_clean_shape):
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
        sep = file.split('.')
        if(other_sep is not None):
            sep_2 = sep[0].split(other_sep)
            id_subj = sep_2[-1]
        else:  
            id_subj = sep[0]
        print ("Reading file of subject: {} ".format(id_subj)) 
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        dataset.add_mesh(int(id_subj),vertices,triangles)
    return dataset


def read_data_folder_csv(src_path, exclude_path,dim,separation=','):
    
    dataset = ShapeDataset(dim)
    
    for file in os.listdir(src_path):
        csv_file = os.path.join(src_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(csv_file,header=None,sep=separation)
            data = df.values[:,0:dim] 
        elif file.endswith('.txt'):
            df = pd.read_csv(csv_file,header=None,sep=separation )
            data = df.values[:,0:dim] 
        else:
            continue
    
        sep = file.split('.')
        id_subj = sep[0]
        print ("Reading file of subject: {} ".format(id_subj)) 
        
        dataset.add_shape(int(id_subj),data)
    return dataset

# Input can be a list or 3d matrix
def write_data(data,id_vec,dest_path,dim):
    
    n_samples = len(id_vec)
    
    if(n_samples==1):
        data = data.reshape(-1,dim)
        id_ = id_vec[0]
        csv_path = os.path.join(dest_path, str(id_)+'.csv')
        np.savetxt(csv_path, data, delimiter=",")
        print('Saved ID {} in {}'.format(str(id_),csv_path))               
    else:
        if(not isinstance(data, list) and (len(data.shape)==2)):
            data = check_matrix(data,dim,int(data.shape[1]/dim),n_samples,'3d')
        
        # Iterate over shapes
        for n_sample in range(n_samples):    
            if(isinstance(data, list)):
                shape = data[n_sample]
            else:
                shape = data[:,:,n_sample]
            # Process
            id_ = id_vec[n_sample]    
            csv_path = os.path.join(dest_path, str(id_)+'.csv')
            np.savetxt(csv_path, shape, delimiter=",")
            print('Saved ID {} in {}'.format(str(id_),csv_path))
