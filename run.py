"""
This script generates 500 shapes from the diffusion learnt on top of the autodecoder.
"""

from tools import * 
from tqdm import tqdm
import os
from metrics import *
import logging

import sys
sys.path.insert(1, '../DeepSDF2')
import deep_sdf

########### Change the variables in this section to run this script ###########
dataset_path = '/mnt/disks/data2/latent_diffusion/03001627_sdfs_500k'

split_path =  '../DeepSDF2/examples/chairs/chairs_train.json'

##############################################################################
############ Generate with Diffusion Model + Autodecoder ####################################
diffusion_model_dt = '' 
diffusion_model_idx = ''
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

# load diffusion model
model_filename = f'./diffusion logs & models/{diffusion_model_dt}/conditional model_{diffusion_model_dt}_{diffusion_model_idx}'
model = ConditionalModel(forward_process_t)
model.load_state_dict(torch.load(model_filename))
model.eval()

# load the decoder and shapecodes
decoder, _ = get_decoder(decoder_model, decoder_epoch=shapecode_epoch) 
shapecode, _, shape_indices_from_decoder = get_shapecode(decoder_model, n_shapes=n_shapes, shapecode_epoch=shapecode_epoch) 

# preprocess shapecode
std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))
shapecode = (shapecode-mean)/std

# generate 500 shapes using diffusion model 
generated_shapes_dir = f'./diffusion logs & models/{diffusion_model_dt}/generated shapes-{diffusion_model_dt}-{diffusion_model_idx}'
if not os.path.isdir(generated_shapes_dir):
    os.mkdir(generated_shapes_dir)
for i in tqdm(range(500)):
    if os.path.isfile(f'{generated_shapes_dir}/{i}.npz'):
        continue
    denoised_code = torch.from_numpy(np.random.normal(0, 1, 256)).float()
    for j in reversed(range(forward_process_t)):
        denoised_code = p_sample(model, denoised_code, j, betas)
    denoised_code = std*denoised_code+mean
    verts, faces = reconstruct_normalized_shape(decoder, denoised_code)

    np.savez(f'{generated_shapes_dir}/{i}.npz', denoised_code=denoised_code.detach().numpy(), verts=verts, faces=faces)

########### Generate by sampling from DeepSDF's Gaussian Distribution #######################
shapecode_epoch = 0
n_shapes = 6272
decoder_model = ''
checkpoint="latest"
decoder, _ = get_deep_sdf_decoder(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint=checkpoint)
shapecode, shapecode_epoch = get_deep_sdf_shapecodes(experiment_directory=f"../DeepSDF2/examples/{decoder_model}", checkpoint=checkpoint)
std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))
generated_shapes_dir = ''
if not os.path.isdir(generated_shapes_dir):
    os.mkdir(generated_shapes_dir)
for i in tqdm(range(500)):
    if os.path.isfile(f'{generated_shapes_dir}/{i}.npz'):
        continue
    code_t = np.random.normal(mean, std, 256)
    with torch.no_grad():
        verts, faces = deep_sdf.mesh.create_mesh(
            decoder,
            code_t,
            "",
            N=256,
            max_batch=int(2 ** 18),
            offset=None,
            scale=None,
        )
    np.savez(f'./{generated_shapes_dir}/{i}', denoised_code=code_t, verts=verts, faces=faces)

########### Generate by sampling from the AutoDecoder's Gaussian Distribution ###############
# load the decoder and shapecodes
shapecode_epoch = 0
n_shapes = 6272
decoder_model = ''
decoder, _ = get_decoder(decoder_model, decoder_epoch=shapecode_epoch) 
shapecode, _, shape_indices_from_decoder = get_shapecode(decoder_model, n_shapes=n_shapes, shapecode_epoch=shapecode_epoch) 
std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))
generated_shapes_dir = ''
if not os.path.isdir(generated_shapes_dir):
    os.mkdir(generated_shapes_dir)
for i in tqdm(range(500)):
    if os.path.isfile(f'{generated_shapes_dir}/{i}.npz'):
        continue
    code_t = np.random.normal(mean, std, 256)
    verts, faces = reconstruct_normalized_shape(decoder, torch.from_numpy(code_t).float())
    np.savez(f'./{generated_shapes_dir}/{i}', denoised_code=code_t, verts=verts, faces=faces)