from tools import * 
from tqdm import tqdm
import os
from metrics import *
import logging

def normalize_point_clouds(pcs, mode):
    if mode is None:
        #logger.info('Will not normalize point clouds.')
        return pcs
    #logger.info('Normalization mode: %s' % mode)
    for i in range(pcs.shape[0]):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

dataset_path = 'data/03001627_sdfs'
split_path =  '../DeepSDF2/examples/chairs/chairs_train.json'
generated_shapes_dir = '../diffusion-point-cloud'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("Generate and run metrics - %(levelname)s - %(message)s")

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)

file_logger_handler = logging.FileHandler(f'{generated_shapes_dir}/metrics.log')
file_logger_handler.setFormatter(formatter)
logger.addHandler(file_logger_handler)


# load 500 original mesh.obj
reference_set_paths = load_files_by_uuid(dataset_path, split_path, 'mesh.npz')
reference_set_paths = np.array(reference_set_paths)
random.seed(888)
inds = random.sample(range(len(reference_set_paths)), 500)
logging.info(f'reference set used: {inds}')
#reference_set_paths = reference_set_paths[inds]
# subsample points from reference set
reference_set = subsample_points(reference_set_paths, 'npz')
ref_pcs = normalize_point_clouds(torch.from_numpy(reference_set), mode='shape_unit')
reference_set_total = ref_pcs.numpy()
reference_set = reference_set_total[inds,:,:]

points = np.load('../diffusion-point-cloud/results/GEN_Ours_chair_1666893191/out.npy')
gen_pcs = normalize_point_clouds(torch.from_numpy(points), mode='shape_unit')
generated_set = gen_pcs[:500,:,:].numpy()
print(generated_set.shape)

# compute metrics
#metric1 = mmd(generated_set, reference_set)
#logging.info(f'MMD: {metric1}')
#metric2 = cov(generated_set, reference_set)
#logging.info(f'COV: {metric2}')
#metric3 = nna(generated_set, reference_set)
#logging.info(f'1NNA: {metric3}')
#metric4 = nnd(generated_set, reference_set_total)
#logging.info(f'NND: {metric4}')

metric1 = mmd(generated_set, reference_set,'emd')
#logging.info(f'MMD EMD: {metric1}')
#metric2 = cov(generated_set, reference_set,'emd')
#logging.info(f'COV EMD: {metric2}')
#metric3 = nna(generated_set, reference_set,'emd')
#logging.info(f'1NNA EMD: {metric3}')
