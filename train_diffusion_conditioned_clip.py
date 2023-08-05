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
from tools import *
import clip
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader, TexturesVertex
)
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from PIL import Image
import wandb

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_in_clip, num_out, n_steps, temb_ch):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.lin_clip = nn.Linear(num_in_clip, num_out)
        # self.embed = nn.Embedding(n_steps, num_out)
        # self.embed.weight.data.uniform_()

        self.temb_proj = torch.nn.Linear(temb_ch,
                                         num_out)
        self.nonlin = torch.nn.SiLU()


    def forward(self, x, y, c):
        out = self.lin(x)
        out = self.temb_proj(self.nonlin(y)) + out
        
        out = self.lin_clip(c) + out
        return out
    
class ConditionalModel(nn.Module):
    def __init__(self, n_steps, ch=32, num_out=64):
        super(ConditionalModel, self).__init__()
        self.ch = ch
        self.temb_ch = ch * 4 # time embedding channel
        self.lin1 = ConditionalLinear(256,512,512, n_steps, self.temb_ch)
        self.lin2 = ConditionalLinear(512,512, 512, n_steps, self.temb_ch)
        self.lin3 = ConditionalLinear(512*2,512, 512, n_steps, self.temb_ch)
        self.lin4 = ConditionalLinear(512,512, 512, n_steps, self.temb_ch)
        self.lin5 = ConditionalLinear(512*2,512, 512, n_steps, self.temb_ch)
        self.lin6 = ConditionalLinear(512,512, 512, n_steps, self.temb_ch)
        self.lin7 = ConditionalLinear(512*2,512, 512, n_steps, self.temb_ch)
        self.lin8 = nn.Linear(512,256)


        # timestep embedding
        self.temb = nn.Sequential(
            torch.nn.Linear(ch,
                            self.temb_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        )
    
    def forward(self, x, y, c): # x, t
        y = get_timestep_embedding(y, self.ch)
        temb = self.temb(y)
        x1 = F.softplus(self.lin1(x, temb, c))
        x = F.softplus(self.lin2(x1, temb, c))
        x = F.softplus(self.lin3(torch.cat((x, x1), dim=1), temb, c))
        x = F.softplus(self.lin4(x, temb, c))
        x = F.softplus(self.lin5(torch.cat((x, x1), dim=1), temb, c))
        x = F.softplus(self.lin6(x, temb, c))
        x = F.softplus(self.lin7(torch.cat((x, x1), dim=1), temb, c))
        return self.lin8(x)

class EMA(object): # Stabilizing training
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow


    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def noise_estimation_loss(model, x_0, x_cond0,clip_cond, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):

    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)) # pick (batch_size//2+1) number of rand integers in bw 0 and n_steps
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long().cuda() # pick other time index symmetrically
    # x0 multiplier
    #t_prev=torch.clip(t-1, min=0)
    a = extract(alphas_bar_sqrt, t, x_0).cuda()
    a1 = extract(alphas_bar_sqrt, t, x_cond0).cuda()
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).cuda()
    e = torch.randn_like(x_0).cuda()
    # e = noise_steps[t, :,:]
    # model input
    x = x_0 * a + e * am1
    x_cond = x_cond0*a1
    output = model(x, t, clip_cond)
    #err = e-output
    #eps1 = e - output
    #eps2 = x_cond - x
    err = x_cond-x+ output*am1
    return (err).square().mean()

def main():
    wandb.init(project="diffusion_on_clip")
    ################ 1. define decoder_type and model_datetime ################
    
    # load the decoder model & latent vectors
    decoder_type = "deep_sdf"  # choose either "SGI" (for the decoder model we implemented) or "deep_sdf" (for the official facebook repo decoder model)
    
    if decoder_type == "SGI": 
        model_datetime = '11012022_133638'
        shapecode, shapecode_epoch, shape_indices = get_shapecode(model_datetime, n_shapes=(6778-500)) # specify shapecode_epoch, shape_indices, n_shapes, random_or_not if needed
    elif decoder_type == "deep_sdf": 
        model_datetime = 'chairs'
        shapecode, shapecode_epoch = get_deep_sdf_shapecodes(experiment_directory=f"../DeepSDF2/examples/{model_datetime}", checkpoint="latest")
    clip_embeddings = get_clip_encodings()  
    ################ 2. define training parameters ################   
    learning_rate = 1e-5
    beta1 = 0.9
    beta2 = 0.9
    eps = 1e-07
    batch_size = 10 # TODO: change this
    n_steps = 30000
    beta_schedule = 'linear' # 'linear', 'quad', 'sigmoid', 'cosine'
    # normalize latent vectors
    #print(shapecode.shape)
    #print(shapecode)
    std = np.std(np.asarray(shapecode))
    mean = np.mean(np.asarray(shapecode))
    shapecode = (shapecode-mean)/std
    #print(shapecode)

    # beta scheduling
    betas = make_beta_schedule(schedule=beta_schedule, n_timesteps=n_steps, start=1e-5, end=1e-2) 
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    print(alphas_bar_sqrt)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    # initialize  the model, optimizer, EMA
    model = ConditionalModel(n_steps).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps)
    ema = EMA(0.9)
    ema.register(model)
    min_loss = 1000
    
    # register logs 
    dt = datetime.now() 
    dt = dt.strftime("%m%d%Y_%H%M%S")
    os.mkdir(f'./diffusion logs & models/{dt}')

    logger = logging.getLogger(f'diffusion logger {dt}')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('Diffusion - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(f'./diffusion logs & models/{dt}/{dt}.log')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    
    # log training parameters
    logger.info(f'Decoder model type = {decoder_type}')
    logger.info(f'Decoder model used = {model_datetime}')
    logger.info(f'Shapecode epoch used = {shapecode_epoch}')
    logger.info(f'# of shapes used for diffusion = {len(shapecode)}')
    if decoder_type == "SGI": 
        logger.info(f'Shape indices = {shape_indices}')
    logger.info(f'# of steps for diffusion forward process = {n_steps}')
    logger.info(f'Learning rate = {learning_rate}')
    logger.info(f'Beta 1 = {beta1}')
    logger.info(f'Beta 2 = {beta2}')
    logger.info(f'Eps = {eps}')
    logger.info(f'Optimzer = {optimizer.__class__.__name__}')
    logger.info(f'Batch size = {batch_size}')
    logger.info(f'Beta schedule = {beta_schedule}')
    logger.info(f'Loss Function = err = x_cond-x+ output*am1')

    # starts training
    for t in range(300000):
        # X is a torch Variable
        permutation = torch.randperm(shapecode.size()[0])
        #permutation_conditioning = torch.randperm(shapecode.size()[0])
        epoch_loss = 0
        for i in range(0, shapecode.size()[0], batch_size):
            # Retrieve current batch
            indices = permutation[i:i+batch_size]
            indices_conditioning = torch.randint(0, shapecode.size()[0], (indices.size()[0],))
            #permutation_conditioning[i:i+batch_size]
            batch_x = shapecode[indices].cuda()
            batch_x_conditioning = shapecode[indices_conditioning].cuda()
            ### transform shapecodes into clip image codes ####
            batch_clip_conditioning = clip_embeddings[indices_conditioning].cuda()
            # Compute the loss.
            # loss = noise_estimation_loss(model, batch_x,e)
            loss = noise_estimation_loss(model, batch_x, batch_x_conditioning, batch_clip_conditioning, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            #loss = noise_estimation_loss(model, batch_x, batch_clip_conditioning, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            #wandb.log({"loss": loss})
            epoch_loss += loss * indices.shape[0]
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # Calling the step function to update the parameters
            optimizer.step()
            # Update the exponential moving average
            ema.update(model)

        epoch_loss = epoch_loss / shapecode.size()[0]
        wandb.log({"loss": epoch_loss})
        #wandb.watch(model)
        #if (t % 100 == 0):
        logger.info(f'{t} loss = {epoch_loss.item()}')
        if epoch_loss < min_loss:
            min_loss = epoch_loss
        #if loss < 0.01:
            torch.save(model.state_dict(), f'./diffusion logs & models/{dt}/clip_conditional model_{dt}_{t}')

if __name__ == "__main__":
    main()
