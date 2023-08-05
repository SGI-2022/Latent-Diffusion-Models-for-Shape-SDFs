from PyTorchEMD.emd import earth_mover_distance
from nvidia_sampler import sample_point_with_mesh_name
from sinkhorn import sinkhorn
import random
import numpy as np
import torch
import kaolin as kal
from tqdm import tqdm
import trimesh
import time
import open3d as o3d

def subsample_points(file_paths, file_type='npz'):
    """
    Sample 2048 points from npz files storing the vertices of shapes.
    
    Input:
    
    file_paths: paths to the npz files.
    file_type: 'obj', 'npz'
    
    Output:
    numpy array Nx2048x3
    """
    
    l = []
    for i in range(len(file_paths)):
        print(file_paths[i])
        file_path = file_paths[i]
        if file_type == 'npz':
            verts = np.load(file_path)['verts']
            faces = np.load(file_path)['faces']

            mesh =  o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
        elif file_type == 'obj':
            mesh=o3d.io.read_triangle_mesh(file_path)
            verts = np.asarray(mesh.vertices)
            
        if verts.shape[0]>2048:
            pcd = mesh.sample_points_poisson_disk(number_of_points=2048)
            points = np.asarray(pcd.points)

        l.append(points)
    
    return np.array(l)


def mmd(generated_set, reference_set, metric="cd"):
    """
    Minimum Matching Distance
    
    Input:
    generated_set, reference_set: numpy array storing the shapes as Nx2048x3. 
    reference_set is a subset of the training_set fixed beforehand.
    
    Use create_objs_from_file() with the file paths to the npz files storing the vertices.
    
    TODO: batching
    """
    print("Start MMD")
    tot = 0
    n_r = reference_set.shape[0]
    n_g = generated_set.shape[0]
    for i in tqdm(range(n_r)):
        shape_r = reference_set[i,:,:]
        shape_r = torch.from_numpy(shape_r).float().cuda()#[None,:,:]
        temp=float('inf')
        
        for j in range(n_g):
            shape_g = generated_set[j,:,:]
            shape_g = torch.from_numpy(shape_g).float().cuda()#[None,:,:]
            if metric=="cd":
                d = kal.metrics.pointcloud.chamfer_distance(shape_r[None,:,:],shape_g[None,:,:])
            else:
                #t = time.time()
                #d = earth_mover_distance(shape_r[None,:,:], shape_g[None,:,:], transpose=False)
                #print(time.time()-t)
                #print("emd: ",d)
                #t = time.time()
                d,_,_ = sinkhorn(shape_r, shape_g, p=2, eps=1e-5, max_iters=500, stop_thresh=1e-3, verbose=False)
                #print(d)
                #print(time.time()-t)
                #print("sinkhorn: ", d)
            if d<temp:
                temp = d
                
        tot+=temp
    
    return tot/n_r


def cov(generated_set, reference_set, metric="cd"):
    """
    Coverage
    
    Input:
    generated_set, reference_set: numpy array storing the shapes as Nx2048x3. 
    reference_set is a subset of the training_set fixed beforehand.
    
    Use create_objs_from_file() with the file paths to the npz files storing the vertices.
    
    TODO: batching
    """
    print("Start COV")
    tot = set()   
    tot.add(-1)
    n_r = reference_set.shape[0]
    n_g = generated_set.shape[0]
    for i in tqdm(range(n_g)):
        shape_g = generated_set[i,:,:]
        shape_g = torch.from_numpy(shape_g).float().cuda()
        temp = float('inf')
        ind = -1
       
        for j in range(n_r):
            shape_r = reference_set[j,:,:]
            shape_r = torch.from_numpy(shape_r).float().cuda()
            if metric=="cd":
                d = kal.metrics.pointcloud.chamfer_distance(shape_r[None,:,:],shape_g[None,:,:])
            else:
                d,_,_ = sinkhorn(shape_r, shape_g, p=2, eps=1e-5, max_iters=500, stop_thresh=1e-3, verbose=False)

            if d<temp:
                temp = d
                ind = j

        tot.add(ind)
    
    return (len(tot)-1)/n_r

def classifier(set1, set2, metric="cd"):
    tot = 0
    n_g = set1.shape[0]
    n_r = set2.shape[0]
    for i in tqdm(range(n_g)):
        shape_g = set1[i,:,:]
        shape_g = torch.from_numpy(shape_g).float().cuda()
        temp = float('inf')
        val = 1
        for j in range(n_g):
            if j!=i:
                shape_g2 = set1[j,:,:]
                shape_g2 = torch.from_numpy(shape_g2).float().cuda()
                if metric=="cd":
                    d = kal.metrics.pointcloud.chamfer_distance(shape_g[None,:,:],shape_g2[None,:,:])
                else:
                    d,_,_ = sinkhorn(shape_g, shape_g2, p=2, eps=1e-5, max_iters=500, stop_thresh=1e-3, verbose=False)

                if d<temp:
                    temp = d

        for j in range(n_r):
            shape_r = set2[j,:,:]
            shape_r = torch.from_numpy(shape_r).float().cuda()
            if metric=="cd":
                d = kal.metrics.pointcloud.chamfer_distance(shape_r[None,:,:],shape_g[None,:,:])
            else:
                d,_,_ = sinkhorn(shape_r, shape_g, p=2, eps=1e-5, max_iters=500, stop_thresh=1e-3, verbose=False)

            if d<temp:
                val = 0
                break
        tot+=val
        
    return tot/(n_r+n_g)

def nna(generated_set, reference_set, metric="cd"):
    """
    1-Nearest Neighbour Classifier
    
    Input:
    generated_set, reference_set: numpy array storing the shapes as Nx2048x3.
    reference_set is a subset of the training_set fixed beforehand.

    Use create_objs_from_file() with the file paths to the npz files storing the vertices.
    
    TODO: batching
    """
    print("Start 1NNA")
    tot = classifier(generated_set, reference_set, metric)
    tot += classifier(reference_set, generated_set, metric)
    
    return tot

def nnd(generated_set, dataset, metric="cd"):
    """
    Overfitting metric
    Same as MMD with the sets considered: generated set and training set
    
    Input:
    generated_set, reference_set: numpy array storing the shapes as Nx2048x3.
    reference_set is the whole training set.
    
    Use create_objs_from_file() with the file paths to the npz files storing the vertices.
    
    TODO: batching
    """
    print("Start NND")
    tot = 0
    n_r = dataset.shape[0]
    n_g = generated_set.shape[0]
    for i in tqdm(range(n_g)):
        shape_g = generated_set[i,:,:]
        shape_g = torch.from_numpy(shape_g).float().cuda()
        temp=float('inf')
        for j in range(n_r):
            shape_r = dataset[j,:,:]
            shape_r = torch.from_numpy(shape_r).float().cuda()
            if metric=="cd":
                d = kal.metrics.pointcloud.chamfer_distance(shape_r[None,:,:],shape_g[None,:,:])
            else:
                d,_,_ = sinkhorn(shape_r, shape_g, p=2, eps=1e-5, max_iters=500, stop_thresh=1e-3, verbose=False)

            if d<temp:
                temp = d

        tot+=temp
    
    return tot/n_g