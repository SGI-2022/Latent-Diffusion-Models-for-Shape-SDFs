from tools import * 
from tqdm import tqdm
import os
from metrics import *
import logging

import sys
sys.path.insert(1, '../DeepSDF2')
import deep_sdf





########### Change the variables in this section to run this script ###########
diffusion_model_dt = '11102022_201043' #planes 
diffusion_model_idx = 7000 # or min_epoch_loss

##############################################################################



# configure logging
generated_shapes_dir = f'./diffusion logs & models/{diffusion_model_dt}/generated shapes-{diffusion_model_dt}-{diffusion_model_idx}'
if not os.path.isdir(generated_shapes_dir):
    os.mkdir(generated_shapes_dir)
logger = logging.getLogger(f'generated shapes-{diffusion_model_dt}-{diffusion_model_idx}')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("Generate and run metrics - %(levelname)s - %(message)s")

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)

file_logger_handler = logging.FileHandler(f'{generated_shapes_dir}/metrics.log')
file_logger_handler.setFormatter(formatter)
logger.addHandler(file_logger_handler)

logging.info(f'Using {diffusion_model_dt}, {diffusion_model_idx}')

# load diffusion log and key training parameters used 
log_filename = f'./diffusion logs & models/{diffusion_model_dt}/{diffusion_model_dt}.log'

print(f'logging at {log_filename}')

with open(log_filename) as f:
    f = f.readlines()
        
decoder_model_type = "SGI" # default value as some earlier logs are missing with this
beta_schedule = "linear"
for line in f: 
    if "Decoder model type" in line:
        decoder_model_type = line.split('=')[1].strip()
    if "# of shapes used for diffusion" in line :
        n_shapes = int(line.split('=')[1].strip())
    if "Decoder model used" in line :
        decoder_model = line.split('=')[1].strip()
        if decoder_model == 'chairs(batch size 64)': # folder name is changed
            decoder_model = 'chairs(50k)'
    if "# of steps for diffusion forward process" in line:
        forward_process_t = int(line.split('=')[1].strip())
    if "Shapecode epoch used" in line:
        shapecode_epoch = int(line.split('=')[1].strip())
    if "Beta schedule = " in line:
        beta_schedule = line.split('=')[1].strip()
betas = make_beta_schedule(schedule=beta_schedule, n_timesteps=int(forward_process_t))
logging.info(f'{decoder_model_type}, {n_shapes}, {decoder_model}, {forward_process_t}, {shapecode_epoch}, {beta_schedule}')

# load diffusion model
model_filename = f'./diffusion logs & models/{diffusion_model_dt}/conditional model_{decoder_model_type}_{diffusion_model_dt}_{diffusion_model_idx}'
model = ConditionalModel(forward_process_t)
model.load_state_dict(torch.load(model_filename))
model.eval()

# load the decoder and shapecodes
if decoder_model_type == "SGI":
    decoder, _ = get_decoder(decoder_model, decoder_epoch=shapecode_epoch, parallel=True) 
    shapecode, _, shape_indices_from_decoder = get_shapecode(decoder_model, n_shapes=n_shapes, shapecode_epoch=shapecode_epoch) 
elif decoder_model_type == "deep_sdf":
    decoder, _ = get_deep_sdf_decoder(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint="latest", parallel=True)
    shapecode, shapecode_epoch = get_deep_sdf_shapecodes(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint="latest")

# preprocess shapecode
std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))
shapecode = (shapecode-mean)/std

# generate 500 shapes using diffusion model 
generate_set_paths = []
for i in tqdm(range(500)):
    generate_set_paths.append(f'{generated_shapes_dir}/{i}.npz')
    if os.path.isfile(f'{generated_shapes_dir}/{i}.npz'):
        continue
    denoised_code = torch.from_numpy(np.random.normal(0, 1, 256)).float()
    for j in reversed(range(forward_process_t)):
        denoised_code = p_sample(model, denoised_code, j, betas)
    denoised_code = std*denoised_code+mean
    if decoder_model_type == "SGI":
        verts, faces = reconstruct_normalized_shape(decoder, denoised_code)
    elif decoder_model_type == "deep_sdf":    
        with torch.no_grad():
            verts, faces = deep_sdf.mesh.create_mesh(
                decoder,
                denoised_code,
                "",
                N=256,
                max_batch=int(2 ** 18),
                offset=None,
                scale=None,
            )
    np.savez(f'{generated_shapes_dir}/{i}.npz', denoised_code=denoised_code.detach().numpy(), verts=verts, faces=faces)


# dataset_path: choose from: 
# 500k chair dataset '/mnt/disks/data2/latent_diffusion/03001627_sdfs_500k'
# 50k chair dataset './data/03001627_sdfs/'
# 500k plane dataset '/mnt/disks/data2/latent_diffusion/02691156_sdfs_500k'

# split_path: choose from: 
# chair split '../DeepSDF2/examples/chairs/chairs_train.json'
# plane split '../DeepSDF2/examples/planes/planes_train.json'

# # load 500 original mesh.obj
# reference_set_paths = load_files_by_uuid(dataset_path, split_path, 'mesh.obj')
# reference_set_paths = np.array(reference_set_paths)
# random.seed(888)
# inds = random.sample(range(len(reference_set_paths)), 500)
# logging.info(f'reference set used: {inds}')
# reference_set_paths = reference_set_paths[inds]

# # # subsample points from generated set
# # generated_set = subsample_points(generate_set_paths, 'npz')

# # # subsample points from reference set
# # reference_set = subsample_points(reference_set_paths, 'obj')

# # # compute metrics
# # metric1 = mmd(generated_set, reference_set)
# # logging.info(f'MMD: {metric1}')
# # metric2 = cov(generated_set, reference_set)
# # logging.info(f'COV: {metric2}')
# # metric3 = nna(generated_set, reference_set)
# # logging.info(f'1NNA: {metric3}')






########### Change the variables in this section to run this script ###########
diffusion_model_dt = '11102022_125942' # chairs, batch size 64
diffusion_model_idx = 9500 # or min_epoch_loss

##############################################################################



# configure logging
generated_shapes_dir = f'./diffusion logs & models/{diffusion_model_dt}/generated shapes-{diffusion_model_dt}-{diffusion_model_idx}'
if not os.path.isdir(generated_shapes_dir):
    os.mkdir(generated_shapes_dir)
logger = logging.getLogger(f'generated shapes-{diffusion_model_dt}-{diffusion_model_idx}')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("Generate and run metrics - %(levelname)s - %(message)s")

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(formatter)
logger.addHandler(logger_handler)

file_logger_handler = logging.FileHandler(f'{generated_shapes_dir}/metrics.log')
file_logger_handler.setFormatter(formatter)
logger.addHandler(file_logger_handler)

logging.info(f'Using {diffusion_model_dt}, {diffusion_model_idx}')

# load diffusion log and key training parameters used 
log_filename = f'./diffusion logs & models/{diffusion_model_dt}/{diffusion_model_dt}.log'

print(f'logging at {log_filename}')

with open(log_filename) as f:
    f = f.readlines()
        
decoder_model_type = "SGI" # default value as some earlier logs are missing with this
beta_schedule = "linear"
for line in f: 
    if "Decoder model type" in line:
        decoder_model_type = line.split('=')[1].strip()
    if "# of shapes used for diffusion" in line :
        n_shapes = int(line.split('=')[1].strip())
    if "Decoder model used" in line :
        decoder_model = line.split('=')[1].strip()
        if decoder_model == 'chairs(batch size 64)': # folder name is changed
            decoder_model = 'chairs(50k)'
    if "# of steps for diffusion forward process" in line:
        forward_process_t = int(line.split('=')[1].strip())
    if "Shapecode epoch used" in line:
        shapecode_epoch = int(line.split('=')[1].strip())
    if "Beta schedule = " in line:
        beta_schedule = line.split('=')[1].strip()
betas = make_beta_schedule(schedule=beta_schedule, n_timesteps=int(forward_process_t))
logging.info(f'{decoder_model_type}, {n_shapes}, {decoder_model}, {forward_process_t}, {shapecode_epoch}, {beta_schedule}')

# load diffusion model
model_filename = f'./diffusion logs & models/{diffusion_model_dt}/conditional model_{decoder_model_type}_{diffusion_model_dt}_{diffusion_model_idx}'
model = ConditionalModel(forward_process_t)
model.load_state_dict(torch.load(model_filename))
model.eval()

# load the decoder and shapecodes
if decoder_model_type == "SGI":
    decoder, _ = get_decoder(decoder_model, decoder_epoch=shapecode_epoch, parallel=True) 
    shapecode, _, shape_indices_from_decoder = get_shapecode(decoder_model, n_shapes=n_shapes, shapecode_epoch=shapecode_epoch) 
elif decoder_model_type == "deep_sdf":
    decoder, _ = get_deep_sdf_decoder(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint="latest", parallel=True)
    shapecode, shapecode_epoch = get_deep_sdf_shapecodes(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint="latest")

# preprocess shapecode
std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))
shapecode = (shapecode-mean)/std

# generate 500 shapes using diffusion model 
generate_set_paths = []
for i in tqdm(range(500)):
    generate_set_paths.append(f'{generated_shapes_dir}/{i}.npz')
    if os.path.isfile(f'{generated_shapes_dir}/{i}.npz'):
        continue
    denoised_code = torch.from_numpy(np.random.normal(0, 1, 256)).float()
    for j in reversed(range(forward_process_t)):
        denoised_code = p_sample(model, denoised_code, j, betas)
    denoised_code = std*denoised_code+mean
    if decoder_model_type == "SGI":
        verts, faces = reconstruct_normalized_shape(decoder, denoised_code)
    elif decoder_model_type == "deep_sdf":    
        with torch.no_grad():
            verts, faces = deep_sdf.mesh.create_mesh(
                decoder,
                denoised_code,
                "",
                N=256,
                max_batch=int(2 ** 18),
                offset=None,
                scale=None,
            )
    np.savez(f'{generated_shapes_dir}/{i}.npz', denoised_code=denoised_code.detach().numpy(), verts=verts, faces=faces)
