"""
This script runs the different metrics for all baselines.
The generated shapes should live in a folder witht he following naming convention ****.obj or ****.npz
"""

from tools import * 
from tqdm import tqdm
import os
from metrics import *
import logging
import torch

import sys
sys.path.insert(1, '../DeepSDF2')
import deep_sdf


####### Variables and helper functions ############
object = "planes"
pvd = False
point_diffusion = False
spaghetti = True
deepsdf_sampling = False
autodecoder_sampling = False
ours = False
ours_deepsdf = False

earthmovers = True
chamferdist = True

#split_path =  '../DeepSDF2/examples/planes/planes_train.json'
#dataset_path = 'data/03001627_sdfs'

generated_shapes_dir1 = '../PVD/generated_pvd_planes'
generated_shapes_dir4 = f'generated_gaussian_deepsdf({object})'
generated_shapes_dir5 = f'generated_gaussian_our_autodecoder({object})'
if object=="chairs":
    generated_shapes_dir3 = '/home/paulaugguerrero_gmail_com/spaghetti/assets/checkpoints/spaghetti_chairs_large/samples/occ'
    generated_shapes_dir6 = './diffusion logs & models/11082022_135018/generated shapes-11082022_135018-8000'
    generated_shapes_dir7 = './diffusion logs & models/11022022_030925/generated shapes-11022022_030925-13000'
else:
    generated_shapes_dir3 = '/home/paulaugguerrero_gmail_com/spaghetti/assets/checkpoints/spaghetti_airplanes/samples/occ'
    generated_shapes_dir6 = './diffusion logs & models/11102022_201043/generated shapes-11102022_201043-7000'
    generated_shapes_dir7 = './diffusion logs & models/11082022_142608/generated shapes-11082022_142608-9000'
    
def norma(pts, input_dim=3):
    all_points_mean = pts.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
    all_points_std = pts.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
    pts = (pts - all_points_mean) / all_points_std
    return pts

def normalize_point_clouds(pcs, mode='shape_bbox'):
    if mode is None:
        #logger.info('Will not normalize point clouds.')
        return pcs
    #logger.info('Normalization mode: %s' % mode)
    for i in range(len(pcs)):
        pc = torch.from_numpy(pcs[i])
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc.numpy()
    return pcs

##### Initialize logger ##########
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("Generate and run metrics - %(levelname)s - %(message)s")

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)

file_logger_handler = logging.FileHandler('all_metrics_final.log')
file_logger_handler.setFormatter(formatter)
logger.addHandler(file_logger_handler)

###### Reference Set ##################
logging.info("********* Loading Reference Set ************")
logging.info(f"********* {object} ************")

# load 500 original mesh.obj
#reference_set_paths = load_files_by_uuid(dataset_path, split_path, 'mesh.obj')
#reference_set_paths = np.array(reference_set_paths)
#reference_set = subsample_points(reference_set_paths, 'obj')
#ref_pcs = normalize_point_clouds(torch.from_numpy(reference_set))
#reference_set_total = ref_pcs.numpy()
#if False:
#    points_tr = []
#    for i in tqdm(range(3545)):
#        path = f'reference_sets/{object}/pvd_train/{i}.npz'
#        verts = np.load(path)['verts']
#        points_tr.append(verts)
#    reference_set_tot = np.array(points_tr)
#    reference_set_tot = normalize_point_clouds(reference_set_tot)
#    print(reference_set_tot.shape)

if ours or ours_deepsdf or point_diffusion or autodecoder_sampling or deepsdf_sampling:
    points = []
    for i in tqdm(range(500)):
        path = f'reference_sets/{object}_test/{i}.npz'
        verts = np.load(path)['verts']
        points.append(verts)
    reference_set = np.array(points)[:,0,:,:]
    reference_set = normalize_point_clouds(reference_set)
    print(reference_set.shape)
    print("Reference Set loaded")

####### Baseline1: PVD ################
if pvd:
    logging.info("********* Start PVD ************")
    points = []
    for i in tqdm(range(500)):
        path = f'reference_sets/{object}/pvd_test/{i}.npz'
        verts = np.load(path)['verts']
        points.append(verts)
    reference_set_pvd = np.array(points)[:,0,:,:]
    reference_set_pvd = normalize_point_clouds(reference_set_pvd)
    print(reference_set_pvd.shape)
    print("Reference Set PVD loaded")
    # Data
    generated_set = []
    for i in range(500):
        #k=np.load(f'{generated_shapes_dir1}/{i}.npz')['verts']
        #generated_set.append(k[random.sample(range(k.shape[0]),2048)])
        generated_set.append(np.load(f'{generated_shapes_dir1}/{i}.npz')['verts'])
    #generated_set = subsample_points(generated_set_paths, 'npz')
    #print(generated_set.shape
    generated_set=np.array(generated_set)[:,0,:,:]
    generated_set = normalize_point_clouds(generated_set)
    print(generated_set.shape)
    #generated_set = generated_set.numpy()
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set_pvd)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set_pvd)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set_pvd)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Test Set Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')
    
    if earthmovers:
        metric1 = mmd(generated_set, reference_set_pvd, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set_pvd, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set_pvd, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')

####### Baseline2: Point Diffusion Cloud #########
if point_diffusion:
    logging.info("********* Start Point Diffusion Cloud ************")
    # Data
    points = np.load('../diffusion-point-cloud/results/GEN_Ours_airplane_1668119637/out.npy')
    generated_set = normalize_point_clouds(points)
    generated_set = generated_set[:500,:,:]
    print(generated_set.shape)
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Test Set Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')

###### Baseline3: Spaghetti ###########
if spaghetti:
    logging.info("********* Start Spaghetti ************")
    points = []
    if object=="chairs":
        n=500
        k = 6755
    else:
        n=456
        k=1775
    for i in tqdm(range(n)):
        path = f'reference_sets/{object}/spaghetti_test/{i}.npz'
        verts = np.load(path)['verts']
        points.append(verts)
    reference_set_spg = np.array(points)[:,0,:,:]
    reference_set_spg = normalize_point_clouds(reference_set_spg)
    print(reference_set_spg.shape)
    print("Reference Set Spaghetti loaded")
    # Data
    generated_set_paths = []
    for i in range(k, k+500):
        generated_set_paths.append(f'{generated_shapes_dir3}/{i}.obj')
    generated_set = subsample_points(generated_set_paths, 'obj')
    generated_set = normalize_point_clouds(generated_set)
    #generated_set = generated_set
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set_spg)
        logging.info(f'MMD Chamfer Distance SPG: {metric1}')
        metric2 = cov(generated_set, reference_set_spg)
        logging.info(f'COV Chamfer Distance SPG: {metric2}')
        metric3 = nna(generated_set, reference_set_spg)
        logging.info(f'1NNA Chamfer Distance SPG: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Test Set Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set_spg, metric="emd")
        logging.info(f'MMD EMD SPG: {metric1}')
        metric2 = cov(generated_set, reference_set_spg, metric="emd")
        logging.info(f'COV EMD SPG: {metric2}')
        metric3 = nna(generated_set, reference_set_spg, metric="emd")
        logging.info(f'1NNA EMD SPG: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')


###### Baseline4: Gaussian Sampling from DeepSDF #######
if deepsdf_sampling:
    logging.info("********* Start DeepSDF Sampling ************")
    # Data
    generated_set_paths = []
    for i in range(500):
        generated_set_paths.append(f'{generated_shapes_dir4}/{i}.npz')
    generated_set = subsample_points(generated_set_paths, 'npz')
    generated_set = normalize_point_clouds(generated_set)
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Test Set Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')


###### Baseline5: Gaussian Sampling from our AutoDecoder ##########
if autodecoder_sampling:
    logging.info("********* Start AutoDecoder Sampling ************")
    # Data
    generated_set_paths = []
    for i in range(500):
        generated_set_paths.append(f'{generated_shapes_dir5}/{i}.npz')
    generated_set = subsample_points(generated_set_paths, 'npz')
    generated_set = normalize_point_clouds(generated_set)
    # Metrics
    if False:
        metric1 = mmd(generated_set, reference_set)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Test Set Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')


##### 3D-LDM (Ours) ################
if ours:
    logging.info("********* Start 3D-LDM ************")
    # Data
    generated_set_paths = []
    for i in range(500):
        generated_set_paths.append(f'{generated_shapes_dir6}/{i}.npz')
    generated_set = subsample_points(generated_set_paths, 'npz')
    generated_set = normalize_point_clouds(generated_set)
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')

##### 3D-LDM (Ours+deepsdf) ################
if ours_deepsdf:
    logging.info("********* Start 3D-LDM DeepSDF ************")
    # Data
    generated_set_paths = []
    for i in range(500):
        generated_set_paths.append(f'{generated_shapes_dir7}/{i}.npz')
    generated_set = subsample_points(generated_set_paths, 'npz')
    generated_set = normalize_point_clouds(generated_set)
    # Metrics
    if chamferdist:
        metric1 = mmd(generated_set, reference_set)
        logging.info(f'MMD Chamfer Distance: {metric1}')
        metric2 = cov(generated_set, reference_set)
        logging.info(f'COV Chamfer Distance: {metric2}')
        metric3 = nna(generated_set, reference_set)
        logging.info(f'1NNA Chamfer Distance: {metric3}')
        #metric4 = nnd(generated_set, reference_set)
        #logging.info(f'NND Chamfer Distance: {metric4}')
        #metric5 = nnd(generated_set, reference_set_tot)
        #logging.info(f'NND Train Set Chamfer Distance: {metric5}')

    if earthmovers:
        metric1 = mmd(generated_set, reference_set, metric="emd")
        logging.info(f'MMD EMD: {metric1}')
        metric2 = cov(generated_set, reference_set, metric="emd")
        logging.info(f'COV EMD: {metric2}')
        metric3 = nna(generated_set, reference_set, metric="emd")
        logging.info(f'1NNA EMD: {metric3}')
        #metric4 = nnd(generated_set, reference_set_total, metric="emd")
        #logging.info(f'NND EMD: {metric4}')

logging.info("************************************")