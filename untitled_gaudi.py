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

# METHOD 3:
# 1. shapes indexed according to the order in file_paths
# 2. assume # of samples of each file is always 50000
# 3. __getitem__ returns lists of n randomly picked shape indices, xyz (1 by 3) points, sdfs
# 4. fast as np.load is run n less times compared to METHOD 1 and 2
# 5. Note that this dataset class takes shape index and it does not iterate over all data points. 
#    Also, no guarantee that each data point is processed only once per epoch

class ChairDataset(Dataset):
    def __init__(self, file_paths, n_points_to_load):
        self.file_paths = file_paths
        self.n_points_per_shape = 50000
        
        for file_path in file_paths:
            training_set = np.load(file_path)
            assert len(training_set['points']) == self.n_points_per_shape, f"{self.n_points_per_shape} data points expected, got: {training_set['points']}"
                
        self.n_points_to_load = n_points_to_load # number of points to load at once 
        
    def __getitem__(self, shape_idx):        
        training_set = np.load(self.file_paths[shape_idx]) # TODO: try mmap_mode='r'
        points = training_set['points']
        sdfs = training_set['sdf']
        
        n_shape_idx = np.full((self.n_points_to_load, 1), shape_idx, dtype=int)
        
        n_rand = random.sample(range(self.n_points_per_shape), self.n_points_to_load) # randomly pick 'n_points_to_load number' of indices  
        
        n_points = points[n_rand]
        
        n_sdf = sdfs[n_rand]
        n_sdf = np.resize(n_sdf, (self.n_points_to_load, 1))
        
        return n_shape_idx, n_points, n_sdf
    
    def __len__(self):
        return len(self.file_paths)
    

def load_files(all_file_or_not, n_files = 0):
    file_paths = []
    main_dir = '../data/03001627_sdfs/'

    if all_file_or_not: # loading all files
        n_files = 0
        for sub_dir in os.scandir(main_dir):
            if sub_dir.is_dir():
                for file in os.listdir(main_dir + sub_dir.name):
                    file_paths.append(main_dir + sub_dir.name + '/' + file) if file.endswith("sdf_samples.npz") else None
            n_files += 1
            
    else: # loading specific # of files
        for sub_dir in os.scandir(main_dir):
            if sub_dir.is_dir():
                for file in os.listdir(main_dir + sub_dir.name):
                    file_paths.append(main_dir + sub_dir.name + '/' + file) if file.endswith("sdf_samples.npz") else None
            if len(file_paths) == n_files:
                break
    
    print(f'total # of files: {n_files}')
    return file_paths

# autodecoder MLP class
# structure reference: deepSDF paper  
class MLP(nn.Module):
    def __init__(self, n_shapes, shape_code_length, n_inner_nodes):
        super(MLP, self).__init__()
        self.shape_code_length = shape_code_length
        self.shape_codes = nn.Embedding(n_shapes, shape_code_length) # shape code as an embedding
        nn.init.normal_(self.shape_codes.weight, mean=0, std=0.01)
        print(self.shape_codes.weight)
        
        self.linear1 = nn.Linear(3 + shape_code_length, n_inner_nodes) # (x, y, z) + shape code 
        self.linear2 = nn.Linear(n_inner_nodes, n_inner_nodes)
        self.linear3 = nn.Linear(n_inner_nodes, n_inner_nodes)
        self.linear4 = nn.Linear(n_inner_nodes, n_inner_nodes - (3 + shape_code_length))
        self.linear5 = nn.Linear(n_inner_nodes, n_inner_nodes)
        self.linear6 = nn.Linear(n_inner_nodes, n_inner_nodes)
        self.linear7 = nn.Linear(n_inner_nodes, n_inner_nodes)
        self.linear8 = nn.Linear(n_inner_nodes, 1) # output a SDF value
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def add_noise_to_shape_codes(self, beta):
        self.shape_codes.weight.data += beta*(torch.rand_like(self.shape_codes.weight) * torch.std(self.shape_codes.weight, unbiased=False))

        
    def forward(self, shape_idx, x):
        shape_code = self.shape_codes(shape_idx.view(1, -1))
        shape_code = shape_code.view(-1, self.shape_code_length)
        shape_code_with_xyz = torch.cat((x, shape_code), dim=1) # concatenate horizontally
        
        out = self.linear1(shape_code_with_xyz)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(torch.cat((out, shape_code_with_xyz), dim=1)) # skip connection
        out = self.relu(out)
        out = self.linear6(out)
        out = self.relu(out)
        out = self.linear7(out)
        out = self.relu(out)
        out = self.linear8(out)

        return out
    
    
def generate_validation_points(file_paths, n_points_per_shape, n_points_to_generate):
    shape_idx = random.sample(range(len(file_paths)), 1)[0] # pick a shape randomly out of 7000 shapes
    n_shape_idx = np.full((n_points_to_generate, 1), shape_idx, dtype=int)
    
    training_set = np.load(file_paths[shape_idx]) 
    points = training_set['points']
    sdfs = training_set['sdf']

    rand = random.sample(range(n_points_per_shape), n_points_to_generate) # pick 1000 points randomly

    return torch.from_numpy(n_shape_idx), torch.from_numpy(points[rand]), torch.from_numpy(sdfs[rand])
    

file_paths = load_files(True)

n_points_per_shape = 50000
n_points_to_load = 2048  # n_points_to_load= n points loaded at once from a single file
batch_size = 10 # batch_size = n shapes loaded in one batch, not n data points
dataset = ChairDataset(file_paths=file_paths, n_points_to_load=n_points_to_load)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) 

# training main loop
n_iters = math.ceil(n_points_per_shape/n_points_to_load) # eg 50000/1024 = 49..
rigor = 1 # to account for randomness
n_iters = n_iters * rigor

n_epochs = 1000 

lr = 1e-5 # originally: 0.001
shape_code_lr = 1e-4
betas = (0.9, 0.9) # origignally: (0.9, 0.999)
eps = 1e-08 * 10 # originally: 1e-08

sigma = 1e2 # regularization term

model = MLP(n_shapes=len(file_paths), shape_code_length=256, n_inner_nodes=512).cuda()
print(model)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam([{'params': model.linear1.weight, 'lr': lr},
                              {'params': model.linear1.bias, 'lr': lr},
                              {'params': model.linear2.weight, 'lr': lr},
                              {'params': model.linear2.bias, 'lr': lr},
                              {'params': model.linear3.weight, 'lr': lr},
                              {'params': model.linear3.bias, 'lr': lr},
                              {'params': model.linear4.weight, 'lr': lr},
                              {'params': model.linear4.bias, 'lr': lr},
                              {'params': model.linear5.weight, 'lr': lr},
                              {'params': model.linear5.bias, 'lr': lr},
                              {'params': model.linear6.weight, 'lr': lr},
                              {'params': model.linear6.bias, 'lr': lr},
                              {'params': model.linear7.weight, 'lr': lr},
                              {'params': model.linear7.bias, 'lr': lr},
                              {'params': model.linear8.weight, 'lr': lr},
                              {'params': model.linear8.bias, 'lr': lr},
                              {'params': model.shape_codes.weight, 'lr': shape_code_lr}], 
                             betas=betas, 
                             eps=eps)


now = datetime.now() 
now = now.strftime("%m%d%Y_%H%M%S")
print(f'datetime now: {now}')

f = open(f'./models/autodecoder_{now}.csv','a') #csv to log loss

for epoch in range(n_epochs):
    
    train_loss = 0.0 # monitor training loss
    
    if epoch != 0:
        with torch.no_grad():
            model.add_noise_to_shape_codes(0.1)
    
    for this_iter in range(n_iters): # iterate over all points for each shape
        for n_idx, n_points, n_sdfs in dataloader: # iterate over n_points_to_load for each shape 
            # load data points
            n_idx = n_idx.cuda()
            n_points = n_points.cuda() 
            n_sdfs = n_sdfs.cuda() 

            n_idx = n_idx.view(-1, 1)
            n_points = n_points.view(-1, 3)
            n_sdfs = n_sdfs.view(-1, 1)

            # rand = random.sample(range(len(n_idx)), len(n_idx)) # randomly pick 'n_points_to_load number' of indices  
            # n_idx = n_idx[rand]
            # n_points = n_points[rand]
            # n_sdfs = n_sdfs[rand]

            # forward
            sdf_pred = model(n_idx, n_points) 
            loss = criterion(torch.clamp(sdf_pred, -0.1, 0.1), torch.clamp(n_sdfs, -0.1, 0.1))
            
            idx = torch.unique(n_idx)
            z = model.shape_codes.weight[idx]
            loss += torch.sum(torch.square(z))/(n_idx.size(0)*(sigma**2)) # add regularization term

            # backward
            optimizer.zero_grad() 
            loss.backward()

            # update
            optimizer.step()

            # update running training loss
            train_loss += loss.item()*n_idx.size(0) # n_idx.size(0) = n_shapes in the batch * n_points_to_load
    
    train_loss_ave = train_loss/(n_iters*(len(dataloader.dataset)*n_points_to_load))        
    print('Epoch: {} \tTraining Loss: {}'.format(epoch+1, train_loss_ave))
    torch.save(model.state_dict(), f'./models/autodecoder_' + now + f'_{epoch}')
    f.write(str(train_loss_ave)+'\n')
    
    with torch.no_grad(): # validation
        for i in range(3): # do validation n times 
            n_points_to_generate=1000
            val_idx, val_points, val_sdfs = generate_validation_points(file_paths, n_points_per_shape, n_points_to_generate)
            val_idx = val_idx.cuda()
            val_idx = val_idx.view(-1, 1)
            
            val_points = val_points.cuda()
            val_points = val_points.view(-1, 3)
            
            val_sdfs = val_sdfs.cuda()
            val_sdfs = val_sdfs.view(-1, 1)
            
            val_sdfs_pred = model(val_idx, val_points)
            val_loss = criterion(torch.clamp(val_sdfs, -0.1, 0.1), torch.clamp(val_sdfs_pred, -0.1, 0.1))
            print(f'Validation: shape {val_idx[0]}, loss for 1000 random points: {val_loss}')
        
f.close()
