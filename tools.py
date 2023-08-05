import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import trimesh
from skimage import measure
import meshplot as mp
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import time
from datetime import timedelta, datetime
import random
import math
import logging
import sys
sys.path.insert(1, '../DeepSDF2/deep_sdf')
import workspace as ws
from PIL import Image
import json


# Some common paths:


# 500k chair DeepSDF model '../DeepSDF2/examples/chairs/'
# 50k chair DeepSDF model  '../DeepSDF2/examples/chairs(50k)/'
# 500k chair dataset '/mnt/disks/data2/latent_diffusion/03001627_sdfs_500k'
# 50k chair dataset './data/03001627_sdfs/'

# 500k plane dataset '/mnt/disks/data2/latent_diffusion/02691156_sdfs_500k'




def get_deep_sdf_decoder(experiment_directory="../DeepSDF2/examples/chairs(batch size 64)", checkpoint="latest", parallel=False):
    """
    returns the pytorch.nn decoder model and the epoch
    """
    specs_filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    if parallel:
        decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, str(checkpoint)+ ".pth")
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    # decoder = decoder.module.cuda()
    decoder.eval()
    
    if checkpoint == "latest":
        checkpoint = torch.load(os.path.join(experiment_directory, ws.logs_filename))["epoch"]

    return decoder, checkpoint

def get_deep_sdf_shapecodes(experiment_directory="../DeepSDF2/examples/chairs(batch size 64)", checkpoint="latest"):
    """
    returns shapecodes ordered according to the train split (ie output from the get_deep_sdf_trainsplit())
    and the epoch
    """
    
    latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)
    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))
    print(ws.logs_filename)
    if checkpoint == "latest":
        checkpoint = torch.load(os.path.join(experiment_directory, ws.logs_filename))["epoch"]
    
    return latent_vectors, checkpoint

def get_deep_sdf_trainsplit(experiment_directory="../DeepSDF2/examples/chairs(batch size 64)"):
    """
    returns a list of shape names (ie uuid) used for the training
    """
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    return train_split['data']['03001627_sdfs']
    
def reconstruct_normalized_shape(decoder, shapecode, N=100, max_batch = 10000):
    """
    decoder: decoder model 
    shapecode: shapecode of a single shape
    N: number of points sampled in x/y/z (thus, total will be N^3 points)
    max_batch = maximum number of subsample points the marching cube can handle
    """
    
    decoder = decoder.cuda()
    shapecode = shapecode.cuda()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(torch.cat((sample_subset, shapecode.repeat(sample_subset.shape[0], 1)), dim=1)).
            squeeze().
            detach().
            cpu())
        head += max_batch

    sdf_values = samples[:, 3]
    numpy_3d_sdf_tensor = sdf_values.reshape(N, N, N).numpy() 

    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates (note x and y are flipped in the output of marching_cubes)
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
    
    return mesh_points, faces
#WARNING: Function is updated
def load_files(main_dir='./data/03001627_sdfs/', filename="mesh.obj", interfix='/',max_n_shapes=6778, n_files = 6778): # if you don't specify n_files, this will load all files
    file_paths = []

    for sub_dir in os.scandir(main_dir):
        if sub_dir.is_dir(): #shape names
            with open(main_dir + sub_dir.name +"/rendering/renderings.txt","r") as f:
                f = f.readlines()            
            nb = random.randint(0,len(f)-1)
            name = f[nb][:-1]

            for file in os.listdir(main_dir + sub_dir.name + interfix):
                file_paths.append(main_dir + sub_dir.name + interfix + file) if file == name else None

        # It should be random, no?
        #if len(file_paths) == n_files:
        #    break
    if n_files<max_n_shapes:
        random.seed(0)
        shape_indices = random.sample(range(max_n_shapes), n_files) # randomly sample n_shapes points out of max_n_shapes
        file_paths = [file_paths[i] for i in shape_indices]
    
    print(f'total # of files: {n_files} out of 6778')
    return file_paths

def load_files_by_uuid(main_dir, split_dir, filename):
    """
    main_dir: path to the dataset folder
    split_dir: path to split json file
    filename: choose from 'surface_samples.npz', 'sdf_samples.npz', or 'mesh.obj'
    """

    with open(split_dir, "r") as f:
        train_split = json.load(f)

    training_set_uuids = []
    for dataset in train_split: # dataset = 'data'
        for class_name in train_split[dataset]: # class_name = '03001627_sdfs'
            for uuid in train_split[dataset][class_name]: # eg. 1006b...70d646
                training_set_uuids.append(uuid)

    file_paths = []
    for uuid in training_set_uuids:
        file_paths.append(os.path.join(main_dir, uuid, filename))

    print(f"loaded {len(file_paths)} number of {filename} files")
    
    return file_paths


def get_clip_encodings(n_files = 6278):
    import clip
    # register logs 
    dt = datetime.now() 
    dt = dt.strftime("%m%d%Y_%H%M%S")
    os.mkdir(f'./diffusion logs & models/clip_paths_{dt}')

    logger = logging.getLogger(f'clip paths {dt}')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('PATH - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(f'./diffusion logs & models/clip_paths_{dt}/{dt}.log')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    image_features = []
    file_paths = load_files('03001627/',".png", interfix='/rendering/',max_n_shapes=6778, n_files=n_files)
    clip_model, preprocess = clip.load('ViT-B/32', "cuda")

    for file_name in file_paths:
        # log training parameters
        logger.info(f'File path usd by CLIP = {file_name}')
        im = Image.open(file_name)
        image_input = preprocess(im).unsqueeze(0).to("cuda")
        with torch.no_grad():
            features = clip_model.encode_image(image_input)
            features = features.cpu()
            features = features.detach().numpy()
        image_features.append(features[0,:])
    return torch.from_numpy(np.array(image_features)).float()
    

def process_decoder_log(datetime):
    with open(os.path.join('./models/', datetime, f'autodecoder_{datetime}.log')) as f:
        f = f.readlines()
    load_all_shapes = 'False'
    for line in f:
        if "load_all_shapes" in line:
            load_all_shapes = line.split(' = ')[1].split(',')[0].strip()
        if load_all_shapes == 'True':
            n_shapes = 6778
        #if "n_shapes" in line:
        #    n_shapes = line.split(' = ')[2].split(',')[0].strip()
        if "Shape codes: size" in line:
            n_shapes = line.split(' = ')[1].split(',')[0].strip()
        if "dropout" in line:
            dropout = line.split('=')[3].split(',')[0].strip()
        if "weight_norm" in line:
            weight_norm = line.split('=')[5].split(',')[0].strip()
        if "use_tanh" in line:
            use_tanh = line.split('=')[6].strip()
        if "n_points_to_load" in line:
            n_points_to_load = line.split(' = ')[2].split(',')[0].strip()
        if "loss" in line and 'Validation' not in line:
            epoch_trained = line.split('-')[2].split('loss')[0].strip()
    
    return int(n_shapes), dropout=="True", weight_norm=="True", use_tanh=="use_tanh", int(epoch_trained)

def get_decoder(datetime, decoder_epoch="latest", parallel=False):
    n_shapes, dropout, weight_norm, use_tanh, epoch_trained = process_decoder_log(datetime)
    
    if decoder_epoch == "latest": 
        decoder_epoch = math.floor(epoch_trained/10) * 10 # use the max trained epoch if not specified

    from train_decoder import MLP
    decoder = MLP(n_shapes, 256, 512, dropout, 0.2, weight_norm, use_tanh)
    if parallel:
        decoder = torch.nn.DataParallel(decoder)
    decoder.load_state_dict(torch.load(f'./models/{datetime}/autodecoder_{datetime}_{decoder_epoch}'))

    return decoder, decoder_epoch
    
def get_shapecode(datetime, shapecode_epoch="latest", shape_indices=None, n_shapes=10, random_or_not=False):
    max_n_shapes, dropout, weight_norm, use_tanh, epoch_trained = process_decoder_log(datetime)
    
    if shapecode_epoch == "latest": 
        shapecode_epoch = math.floor(epoch_trained/10) * 10 # use the max trained epoch if not specified

    shapecode = torch.nn.Embedding(max_n_shapes, 256)
    shapecode.load_state_dict(torch.load(f'./models/{datetime}/shapecode_{datetime}_{shapecode_epoch}'))
    shapecode=shapecode.weight.data.detach()
    
    if shape_indices is None:
        if random_or_not:
            random.seed()
        else:
            random.seed(0)
        shape_indices = random.sample(range(max_n_shapes), n_shapes) # randomly sample n_shapes points out of max_n_shapes
    
    shapecode = shapecode[torch.tensor(shape_indices)]        
    
    return shapecode, shapecode_epoch, shape_indices

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2, offset=0.007):
    print(schedule)
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine": # Eqn 17 from Improved Denoising Diffusion Probabilistic Models
        t = torch.linspace(0, n_timesteps, n_timesteps+1)
        alphas_prod = torch.cos((t/n_timesteps + offset)/(1 + offset) * (math.pi/2))**2 #TODO: tune 'offset'
        alphas_prod = alphas_prod/alphas_prod[0]    
        betas = 1 - (alphas_prod[1:]/alphas_prod[:-1])
        betas = torch.clip(betas, start, end) # torch.clip(betas, 0.001, 0.9999)
    return betas

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, temb_ch):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        # self.embed = nn.Embedding(n_steps, num_out)
        # self.embed.weight.data.uniform_()

        self.temb_proj = torch.nn.Linear(temb_ch,
                                         num_out)
        self.nonlin = torch.nn.SiLU()


    def forward(self, x, y):
        out = self.lin(x)

        out = self.temb_proj(self.nonlin(y)) + out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, n_steps, ch=32, num_out=64):
        super(ConditionalModel, self).__init__()
        self.ch = ch
        self.temb_ch = ch * 4 # time embedding channel
        self.lin1 = ConditionalLinear(256,512, n_steps, self.temb_ch)
        self.lin2 = ConditionalLinear(512, 512, n_steps, self.temb_ch)
        self.lin3 = ConditionalLinear(512*2, 512, n_steps, self.temb_ch)
        self.lin4 = ConditionalLinear(512, 512, n_steps, self.temb_ch)
        self.lin5 = ConditionalLinear(512*2, 512, n_steps, self.temb_ch)
        self.lin6 = ConditionalLinear(512, 512, n_steps, self.temb_ch)
        self.lin7 = ConditionalLinear(512*2, 512, n_steps, self.temb_ch)
        self.lin8 = nn.Linear(512,256)


        # timestep embedding
        self.temb = nn.Sequential(
            torch.nn.Linear(ch,
                            self.temb_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        )

    
    def forward(self, x, y): # x, t
        y = get_timestep_embedding(y, self.ch)
        temb = self.temb(y)
        x1 = F.softplus(self.lin1(x, temb))
        x = F.softplus(self.lin2(x1, temb))
        x = F.softplus(self.lin3(torch.cat((x, x1), dim=1), temb))
        x = F.softplus(self.lin4(x, temb))
        x = F.softplus(self.lin5(torch.cat((x, x1), dim=1), temb))
        x = F.softplus(self.lin6(x, temb))
        x = F.softplus(self.lin7(torch.cat((x, x1), dim=1), temb))
        return self.lin8(x)

def p_sample(model, x, t, betas):
    alphas = 1 - betas
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return (sample)

def q_sample(x_0, t, betas, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
        
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)
