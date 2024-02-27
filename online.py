import os
import random
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
random.seed(0)
from pathlib import Path
import matplotlib.pyplot as plt
from torch import log
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from dataloader.cobot_loader import load_cobot_data, Cdata_generator, CConfig

from models.DDNet_Original import DDNet_Original as DDNet
from utils import makedir
import sys
import time
import numpy as np
import logging
random.seed(0)

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
POSE_PATH = r"pose_new_v2"

# read txt line by line
import os
from numpy.lib.stride_tricks import as_strided

def create_windows(np_pose, window_size=60):
    stride = np_pose.strides[0]
    shape = (np_pose.shape[0] - window_size + 1, window_size) + np_pose.shape[1:]
    strides = (stride,) + np_pose.strides
    windows = as_strided(np_pose, shape=shape, strides=strides)
    return windows
pose_filename_list = os.listdir(POSE_PATH)
def read_txt(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content
def predict (model, device, test_loader):
    model.eval()
    pred_list = list()
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output, _features = model(M, P)
            pred = output.argmax(dim=1, keepdim=True)
            # to list
            pred = pred.cpu().numpy().tolist()
            pred_list.extend(pred)
    return pred_list
        

    
device = torch.device("cuda")
kwargs = {'batch_size': 64}
kwargs.update({'num_workers': 12,
                    'pin_memory': True,
                    'shuffle': True},)

Config = CConfig()
data_generator = Cdata_generator
load_data = load_cobot_data
clc_num = Config.clc_num


# 

# In[11]:


C = Config


# In[12]:

exp = 'only_ddnet_new'
run_name = 'only_ddnet'
import os

Net = DDNet(C.frame_l, C.joint_n, C.joint_d,
            C.feat_d, C.filters, clc_num, run_name)
model = Net.to(device)

# if exp not exist, create it   
if not os.path.exists(f'online_results'):
    os.makedirs(f'online_results')
if not os.path.exists(f'online_results/{exp}'):
    os.makedirs(f'online_results/{exp}')

model.load_state_dict(torch.load(f'/home/dev/DD-Net-Pytorch/experiments/{exp}/model.pt'))
model.eval()

test_list = read_txt('test_person.txt')
count = 0
for name in pose_filename_list:
    if not name[:name.rindex('_')] in test_list:
        continue
    print(name)
    
    np_pose = np.load(os.path.join(POSE_PATH, name))
    sample_list = create_windows(np_pose)
        
    le = None
    sample_dict = {
        'pose': sample_list,
        'label': [0]*len(sample_list)
    } 
    X_0_t, X_1_t, Y_t = data_generator(sample_dict, C, le)
    X_0_t = torch.from_numpy(X_0_t).type('torch.FloatTensor')
    X_1_t = torch.from_numpy(X_1_t).type('torch.FloatTensor')
    Y_t = torch.from_numpy(Y_t).type('torch.LongTensor')

    
    

    testset = torch.utils.data.TensorDataset(X_0_t, X_1_t, Y_t)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=16)
    pred_results = predict(model, device, test_loader)
    # write to txt 
    with open(f"online_results/{exp}/{name[:-4]}.txt", "w") as f:
        for pred in pred_results:
            f.write(str(pred[0]) + '\n')
    # save to npy
    np.save(f"online_results/{exp}/{name[:-4]}.npy", pred_results)


        