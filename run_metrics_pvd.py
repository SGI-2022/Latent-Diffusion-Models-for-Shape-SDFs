from tools import * 
from tqdm import tqdm
import os
from metrics import *
import logging

def norma(pts, input_dim=3):
    all_points_mean = pts.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
    all_points_std = pts.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
    pts = (pts - all_points_mean) / all_points_std
    return pts

dataset_path = 'data/03001627_sdfs'
split_path =  '../DeepSDF2/examples/chairs/chairs_train.json'
generated_shapes_dir = '../PVD/generated'

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
ref_pcs = norma(torch.from_numpy(reference_set))
reference_set_total = ref_pcs.numpy()
reference_set = reference_set_total[inds,:,:]
print(reference_set.shape)

generated_set_paths = []
for i in range(500):
    generated_set_paths.append(f'{generated_shapes_dir}/{i}.npz')
generated_set = subsample_points(generated_set_paths, 'npz')
generated_set = norma(torch.from_numpy(generated_set))
generated_set = generated_set.numpy()[:,0,:,:]
print(generated_set.shape)

# compute metrics
metric1 = mmd(generated_set, reference_set)
logging.info(f'MMD: {metric1}')
metric2 = cov(generated_set, reference_set)
logging.info(f'COV: {metric2}')
metric3 = nna(generated_set, reference_set)
logging.info(f'1NNA: {metric3}')
metric4 = nnd(generated_set, reference_set_total)
logging.info(f'NND: {metric4}')

#metric1 = mmd(generated_set, reference_set,'emd')
#logging.info(f'MMD EMD: {metric1}')
#metric2 = cov(generated_set, reference_set,'emd')
#logging.info(f'COV EMD: {metric2}')
#metric3 = nna(generated_set, reference_set,'emd')
#logging.info(f'1NNA EMD: {metric3}')
