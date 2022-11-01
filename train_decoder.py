import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import trimesh
from skimage import measure
import meshplot as mp
from torch.utils.data import DataLoader, Dataset
import os
import time
from datetime import timedelta, datetime
import random
import math
from itertools import chain as chain
import torch.nn.functional as F
import logging
import json


# Dataset Method 3:
# 1. shapes indexed according to the order in file_paths
# 2. assume # of samples of each file is always 50000
# 3. __getitem__ returns lists of n randomly picked shape indices, xyz (1 by 3) points, sdfs
# 4. fast as np.load is run n less times compared to METHOD 1 and 2
# 5. Note that this dataset class takes shape index and it does not iterate over all data points. 
#    Also, no guarantee that each data point is processed only once per epoch
class ChairDataset(Dataset):
    def __init__(self, file_paths, n_points_per_shape, n_points_to_load, load_pos_neg):
        self.file_paths = file_paths
        self.n_points_per_shape = n_points_per_shape
        
        for file_path in file_paths:
            training_set = np.load(file_path)
            assert len(training_set['points']) == self.n_points_per_shape, \
                    f"{self.n_points_per_shape} data points expected, got: {training_set['points']}"
                
        self.n_points_to_load = n_points_to_load # number of points to load at once 
        self.load_pos_neg = load_pos_neg
        
    def __getitem__(self, shape_idx):
        if self.load_pos_neg:
            self.n_points_to_load = math.floor(self.n_points_to_load/2) * 2

            pos_training_set = np.load(os.path.join(self.file_paths[shape_idx].rsplit('/', 1)[0], 'pos_sdf_samples.npz'))
            pos_points = pos_training_set['points']
            pos_sdfs = pos_training_set['sdf']
            assert int(self.n_points_to_load/2) < len(pos_points), 'not enough positive sdf points'
            n_rand = random.sample(range(len(pos_points)), int(self.n_points_to_load/2))
            n_pos_points = pos_points[n_rand]
            n_pos_sdfs = pos_sdfs[n_rand]

            neg_training_set = np.load(os.path.join(self.file_paths[shape_idx].rsplit('/', 1)[0], 'neg_sdf_samples.npz'))
            neg_points = neg_training_set['points']
            neg_sdfs = neg_training_set['sdf']
            assert int(self.n_points_to_load/2) < len(neg_points), 'not enough negative sdf points'
            n_rand = random.sample(range(len(neg_points)), int(self.n_points_to_load/2))
            n_neg_points = neg_points[n_rand]
            n_neg_sdfs = neg_sdfs[n_rand]
            
            n_shape_idx = np.full((self.n_points_to_load, 1), shape_idx, dtype=int)
            n_points = np.concatenate((n_pos_points, n_neg_points), axis = 0)
            n_sdfs = np.concatenate((n_pos_sdfs, n_neg_sdfs), axis = 0)
            n_sdfs = np.resize(n_sdfs, (self.n_points_to_load, 1))
            
        else:
            training_set = np.load(self.file_paths[shape_idx]) # TODO: try mmap_mode='r'
            points = training_set['points']
            sdfs = training_set['sdf']

            n_shape_idx = np.full((self.n_points_to_load, 1), shape_idx, dtype=int)

            # randomly pick 'n_points_to_load number' of indices  
            n_rand = random.sample(range(self.n_points_per_shape), self.n_points_to_load) 
            n_points = points[n_rand]
            n_sdfs = sdfs[n_rand]
            n_sdfs = np.resize(n_sdfs, (self.n_points_to_load, 1))

        return n_shape_idx, n_points, n_sdfs
    
    def __len__(self):
        return len(self.file_paths)
    
    
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

    
# autodecoder MLP class (architecture reference: deepSDF paper)
class MLP(nn.Module): #TODO: if dropout, dropout_prob, ...
    def __init__(self, n_shapes, shape_code_length, n_inner_nodes, 
                 dropout, dropout_prob, weight_norm, use_tanh):
        
        super(MLP, self).__init__()
        
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.use_tanh = use_tanh

        if weight_norm:
            self.linear1 = nn.utils.weight_norm(nn.Linear(3 + shape_code_length, n_inner_nodes)) # (x, y, z) + shape code 
            self.linear2 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear3 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear4 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes - (3 + shape_code_length)))
            self.linear5 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear6 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear7 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear8 = nn.utils.weight_norm(nn.Linear(n_inner_nodes, 1)) # output a SDF value
        else:
            self.linear1 = (nn.Linear(3 + shape_code_length, n_inner_nodes)) # (x, y, z) + shape code 
            self.linear2 = (nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear3 = (nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear4 = (nn.Linear(n_inner_nodes, n_inner_nodes - (3 + shape_code_length)))
            self.linear5 = (nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear6 = (nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear7 = (nn.Linear(n_inner_nodes, n_inner_nodes))
            self.linear8 = (nn.Linear(n_inner_nodes, 1)) # output a SDF value

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, shape_code_with_xyz):        
        out = self.relu(self.linear1(shape_code_with_xyz))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear2(out))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear3(out))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear4(out))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear5(torch.cat((out, shape_code_with_xyz), dim=1))) # skip connection
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear6(out))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        out = self.relu(self.linear7(out))
        if self.dropout:
            out = F.dropout(out, p=self.dropout_prob)
        
        if self.use_tanh:
            return self.tanh(self.linear8(out))
        else:
            return self.linear8(out)
    
    
def generate_validation_points(file_paths, n_points_per_shape, n_points_to_generate):
    shape_idx = random.sample(range(len(file_paths)), 1)[0] # pick a shape randomly out of 7000 shapes
    n_shape_idx = np.full((n_points_to_generate, 1), shape_idx, dtype=int)
    
    training_set = np.load(file_paths[shape_idx]) 
    points = training_set['points']
    sdfs = training_set['sdf']

    rand = random.sample(range(n_points_per_shape), n_points_to_generate) # pick 1000 points randomly

    return torch.from_numpy(n_shape_idx), torch.from_numpy(points[rand]), torch.from_numpy(sdfs[rand])
    

def configure_logging(log_level, log_filepath):
    logger = logging.getLogger()
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level == "QUIET":
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    file_logger_handler = logging.FileHandler(log_filepath)
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)


def main_function(n_points_per_shape, 
                  n_points_to_load, batch_size, 
                  n_iters, n_epochs, lr, shape_code_lr, lr_schedule, sigma, use_reg,
                  shape_code_length, n_inner_nodes, dropout, dropout_prob, weight_norm, use_tanh,
                  load_pos_neg,
                  main_dir, data_dir, split_dir, num_data_loader_threads, continue_training=None):

    logging.info(f"dataset path = {data_dir}, training split path = {split_dir}, num_data_loader_threads = {num_data_loader_threads}, load_pos_neg = {load_pos_neg}")
    file_paths = load_files_by_uuid(data_dir, split_dir, 'sdf_samples.npz')

    logging.info(f"decoder network: shape_code_length = {shape_code_length}, " +
                 f"n_inner_nodes = {n_inner_nodes}, " + 
                 f"dropout = {dropout}, " + 
                 f"dropout_prob= {dropout_prob}, " +
                 f"weight_norm= {weight_norm}, " + 
                 f"use_tanh={use_tanh}")

    model = MLP(n_shapes=len(file_paths), 
                shape_code_length=shape_code_length,
                n_inner_nodes=n_inner_nodes, 
                dropout=dropout,
                dropout_prob=dropout_prob,
                weight_norm=weight_norm,
                use_tanh=use_tanh).cuda()
    
    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = ChairDataset(file_paths=file_paths, n_points_per_shape=n_points_per_shape, n_points_to_load=n_points_to_load, load_pos_neg=load_pos_neg)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_data_loader_threads) 


    shape_codes = nn.Embedding(len(file_paths), shape_code_length).cuda() # shape code as an embedding
    shape_codes_mean = 0
    shape_codes_mean_std = 1.0/math.sqrt(shape_code_length)
    torch.nn.init.normal_(shape_codes.weight.data, mean=shape_codes_mean, std=shape_codes_mean_std)
    logging.info(f"Shape codes: size = {shape_codes.weight.data.shape[0]}, {shape_codes.weight.data.shape[1]}, " + 
                 f"mean = {torch.mean(shape_codes.weight.data)}, " + 
                 f"std = {torch.std(shape_codes.weight.data)}")

    logging.info(f"Hyperparameters: n_points_per_shape = {n_points_per_shape}, " +
                 f"n_points_to_load = {n_points_to_load}, " +
                 f"batch_size = {batch_size}, " +
                 f"n_iters = {n_iters}, " +
                 f"n_epochs = {n_epochs}, " + 
                 f"lr = {lr}, " + 
                 f"shape_code_lr = {shape_code_lr}, " +
                 f"lr_schedule = {lr_schedule}, " +
                 f"sigma = {sigma}, " + 
                 f"use_reg = {use_reg}")

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':lr}, 
                                  {'params': shape_codes.parameters(), 'lr': shape_code_lr}])

    if continue_training is not None:
        logging.info(f"Continue training from {continue_training}")
        continue_training_datetime = continue_training.rsplit('_', 1)[0].strip()
        model.load_state_dict(torch.load(f'./models/{continue_training_datetime}/autodecoder_{continue_training}'))
        shape_codes.load_state_dict(torch.load(f'./models/{continue_training_datetime}/shapecode_{continue_training}'))


    ########## training starts here ##########
    logging.info("starts training")

    training_start_time = datetime.now() 

    for epoch in range(n_epochs):

        train_loss = 0.0

        model.train()

        # learning rate schedule
        if lr_schedule:
            optimizer.param_groups[0]['lr'] = lr * (0.5 ** (epoch // 500))
            optimizer.param_groups[1]['lr'] = shape_code_lr * (0.5 ** (epoch // 500))

        for this_iter in range(n_iters): # iterate over all points for each shape
            for n_idx, n_points, n_sdfs in dataloader: # iterate over n_points_to_load for each shape 
                # load data points
                n_idx = n_idx.cuda()
                n_points = n_points.cuda() 
                n_sdfs = n_sdfs.cuda() 

                n_idx = n_idx.view(-1, 1)            
                n_points = n_points.view(-1, 3)
                n_sdfs = n_sdfs.view(-1, 1)

                # forward
                n_shape_codes = shape_codes(n_idx.view(-1))
                n_shape_codes_with_xyz = torch.cat((n_points, n_shape_codes), dim=1) # concatenate horizontally
                sdf_pred = model(n_shape_codes_with_xyz) 
                loss = criterion(torch.clamp(sdf_pred, -0.1, 0.1), torch.clamp(n_sdfs, -0.1, 0.1).cuda())

                if epoch >= 100 and use_reg:
                    z_norm = torch.sum(torch.norm(n_shape_codes, dim=1))
                    loss += (z_norm/(n_idx.size(0)*(sigma**2))).cuda() # add regularization term

                # backward
                optimizer.zero_grad() 
                loss.backward()

                # update
                optimizer.step()

                # update running training loss
                train_loss += loss.item()*n_idx.size(0) 
    
        train_loss_ave = train_loss/(n_iters*(len(dataloader.dataset)*n_points_to_load))        
        logging.info(f"{epoch} loss = {train_loss_ave}")
        if epoch%10 == 0: 
            torch.save(model.state_dict(), main_dir + '/autodecoder_' + now + f'_{epoch}')
            torch.save(shape_codes.state_dict(), main_dir + '/shapecode_' + now + f'_{epoch}')

        with torch.no_grad(): # validation
            model.eval() #TODO: eval() doesn't work for F.dropout()
            for i in range(1): # do validation n times 
                n_points_to_generate=1000
                val_idx, val_points, val_sdfs = generate_validation_points(file_paths, n_points_per_shape, n_points_to_generate)
                val_idx = val_idx.cuda()
                val_idx = val_idx.view(-1, 1)

                val_points = val_points.cuda()
                val_points = val_points.view(-1, 3)

                val_sdfs = val_sdfs.cuda()
                val_sdfs = val_sdfs.view(-1, 1)

                val_shape_codes = shape_codes(val_idx.view(-1))
                val_shape_codes_with_xyz = torch.cat((val_points, val_shape_codes), dim=1) # concatenate horizontally
                val_sdfs_pred = model(val_shape_codes_with_xyz)
                val_loss = criterion(torch.clamp(val_sdfs, -0.1, 0.1), torch.clamp(val_sdfs_pred, -0.1, 0.1))
                logging.info(f'Validation: shape {val_idx[0]}, loss for 1000 random points: {val_loss}')
            model.train()
        
    training_end_time = datetime.now()
    elapsed_sec = training_end_time - training_start_time
    logging.info(f"Completed training for {now}")
    logging.info(f"Training duration = {elapsed_sec}")

                     
if __name__ == "__main__":
    now = datetime.now()
    now = now.strftime("%m%d%Y_%H%M%S")
    main_dir = f'./models/{now}'
    os.mkdir(main_dir)
    configure_logging("DEBUG", main_dir + f'/autodecoder_{now}.log')
    main_function(n_points_per_shape = 500000,
                 n_points_to_load = 16384, ####  # n points loaded at once from a single file
                 batch_size = 64, #### # batch_size = n shapes loaded in one batch, not n data points
                  
                 # n_iters = math.ceil(n_points_per_shape/n_points_to_load) # eg 50000/1024 = 49..
                 # rigor = 1 # to account for randomness
                 # n_iters = n_iters * rigor
                 n_iters = 1,
                 n_epochs = 3000 + 1, ####

                 lr = 1e-4, 
                 shape_code_lr = 1e-3,
                 lr_schedule = True,
                 # betas = (0.9, 0.9), # origignally: (0.9, 0.999)
                 # eps = 1e-08 * 10, # originally: 1e-08

                 sigma = 1e2, # regularization term
                 use_reg = True,

                 shape_code_length = 256,
                 n_inner_nodes = 512,
                 dropout = False, 
                 dropout_prob = 0.2, 
                 weight_norm = False, 
                 use_tanh = False,
                                  
                 load_pos_neg = True,
                 
                 main_dir = main_dir,
                 
                 data_dir =  '/mnt/disks/data2/latent_diffusion/03001627_sdfs_500k/',
                 split_dir = '../DeepSDF2/examples/chairs(batch size 8)/chairs_train.json',
                 num_data_loader_threads = 16)

# <chairs>
# data_dir = '/mnt/disks/data2/latent_diffusion/03001627_sdfs_500k/'
# split_dir = '../DeepSDF2/examples/chairs(batch size 8)/chairs_train.json'

# <planes>
# data_dir = '/mnt/disks/data2/latent_diffusion/02691156_sdfs_500k/'
# split_dir = '../DeepSDF2/examples/planes(new json)/planes_train.json'


    