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


def process_decoder_log(datetime):
    with open(os.path.join('./models/', datetime, f'autodecoder_{datetime}.log')) as f:
        f = f.readlines()

    for line in f:
        if "load_all_shapes" in line:
            load_all_shapes = line.split(' = ')[1].split(',')[0].strip()
        if "n_shapes" in line:
            n_shapes = line.split(' = ')[2].split(',')[0].strip()
        if load_all_shapes == 'True':
            n_shapes = 6778
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

def get_decoder(datetime, decoder_epoch=None):
    n_shapes, dropout, weight_norm, use_tanh, epoch_trained = process_decoder_log(datetime)
    
    if decoder_epoch is None: 
        decoder_epoch = math.floor(epoch_trained/10) * 10 # use the max trained epoch if not specified

    from train_decoder import MLP
    decoder = MLP(n_shapes, 256, 512, dropout, 0.2, weight_norm, use_tanh)
    decoder.load_state_dict(torch.load(f'./models/{datetime}/autodecoder_{datetime}_{decoder_epoch}'))

    return decoder, decoder_epoch
    
def get_shapecode(datetime, shapecode_epoch=None, shape_indices=None, n_shapes=10, random_or_not=False):
    max_n_shapes, dropout, weight_norm, use_tanh, epoch_trained = process_decoder_log(datetime)
    
    if shapecode_epoch is None: 
        shapecode_epoch = math.floor(epoch_trained/10) * 10 # use the max trained epoch if not specified

    shapecode = torch.nn.Embedding(max_n_shapes, 256)
    shapecode.load_state_dict(torch.load(f'./models/{datetime}/shapecode_{datetime}_{shapecode_epoch}'))
    shapecode=shapecode.weight.data.detach()
    print(shapecode.shape)
    
    if shape_indices is None:
        if random_or_not:
            random.seed()
        else:
            random.seed(0)
        shape_indices = random.sample(range(max_n_shapes), n_shapes) # randomly sample n_shapes points out of max_n_shapes
    
    shapecode = shapecode[torch.tensor(shape_indices)]        
    
    return shapecode, shapecode_epoch, shape_indices

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
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

def p_sample(model, x, t):
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

# def noise_estimation_loss(model, x_0,noise_steps):
def noise_estimation_loss(model, x_0):

    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)) # pick (batch_size//2+1) number of rand integers in bw 0 and n_steps
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long().cuda() # pick other time index symmetrically
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0).cuda()
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).cuda()
    e = torch.randn_like(x_0).cuda()
    # e = noise_steps[t, :,:]
    # model input
    x = x_0 * a + e * am1
    output = model(x, t)
    return (e - output).square().mean()

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

        
        
def main():

    # load the decoder model & latent vectors
    model_datetime = '09152022_190445'
    decoder, decoder_epoch = get_decoder(model_datetime) # specify decoder_epoch if needed
    shapecode, shapecode_epoch, shape_indices = get_shapecode(model_datetime, n_shapes=(6778-500)) # specify shapecode_epoch, shape_indices, n_shapes, random_or_not if needed
    print(shapecode)
    print(shapecode.shape)

    # normalize latent vectors
    std = np.std(np.asarray(shapecode))
    mean = np.mean(np.asarray(shapecode))
    shapecode = (shapecode-mean)/std
    print(shapecode)

    # beta scheduling
    n_steps = 30000
    betas = make_beta_schedule(schedule='linear', n_timesteps=n_steps, start=1e-5, end=1e-2)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    print(alphas_bar_sqrt)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    # define training parameters
    model = ConditionalModel(n_steps).cuda()
    learning_rate = 1e-5
    beta1 = 0.9
    beta2 = 0.9
    eps = 1e-07

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps)

    ema = EMA(0.9)
    ema.register(model)

    batch_size = 10 # TODO: change this

    dt = datetime.now() 
    dt = dt.strftime("%m%d%Y_%H%M%S")
    os.mkdir(f'./diffusion logs & models/{dt}')

    min_loss = 1000

    logger = logging.getLogger(f'diffusion logger {dt}')
    logger.setLevel(logging.DEBUG)

    fomatter = logging.Formatter('Diffusion - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(f'./diffusion logs & models/{dt}/{dt}.log')
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    logger.info(f'Decoder model used = {model_datetime}')
    logger.info(f'Decoder epoch used = {decoder_epoch}')
    logger.info(f'Shapecode epoch used = {shapecode_epoch}')
    logger.info(f'# of shapes used for diffusion = {len(shape_indices)}')
    logger.info(f'Shape indices = {shape_indices}')
    logger.info(f'# of steps for diffusion forward process = {n_steps}')
    logger.info(f'Learning rate = {learning_rate}')
    logger.info(f'Beta 1 = {beta1}')
    logger.info(f'Beta 2 = {beta2}')
    logger.info(f'Eps = {eps}')
    logger.info(f'Optimzer = {optimizer.__class__.__name__}')
    logger.info(f'Batch size = {batch_size}')

    # starts training
    for t in range(300000):
        # X is a torch Variable
        permutation = torch.randperm(shapecode.size()[0])
        epoch_loss = 0
        for i in range(0, shapecode.size()[0], batch_size):
            # Retrieve current batch
            indices = permutation[i:i+batch_size]
            batch_x = shapecode[indices].cuda()
            # Compute the loss.
            # loss = noise_estimation_loss(model, batch_x,e)
            loss = noise_estimation_loss(model, batch_x)
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
        if (t % 100 == 0):
            logger.info(f'{t} loss = {loss.item()}')
            if loss < min_loss:
                min_loss = loss
            if loss < 0.01:
                torch.save(model.state_dict(), f'./diffusion logs & models/{dt}/conditional model_{dt}_{t}')


if __name__ == "__main__":
    main()