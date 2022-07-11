from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
import pickle
import time
import random
import math

from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from utils import *
from model import *
from dataloader import *

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
DATASET_PATH = PROJECT_PATH + "/dataset/"

if __name__ == '__main__':
    init_time = time.time() 

    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--exp_name', default='clustering_220706_2', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM_feature', help='dataset name')
    parser.add_argument('--data_type', default='train', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=128, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--feature_dim', default=5, help='interval between images in trajectory')
    parser.add_argument('--latent_dim', default=40, help='interval between images in trajectory')
    parser.add_argument('--learning_rate', default=5e-5, help='interval between images in trajectory')
    parser.add_argument('--n_neg', default=64, help='interval between images in trajectory')
    parser.add_argument('--temperature', default=0.05, help='interval between images in trajectory')
    

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/" + args.data_type + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_PATH = DATA_PATH + args.data_type + "/"
    MODEL_PATH = EXP_PATH + "best.pt"

    with open(DATASET_PATH + "cluster_id.pkl", 'rb') as f:
        cluster_id = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = torch.load(MODEL_PATH)
    # model = MLP(args.rollout * args.feature_dim, args.latent_dim, [128, 64, 32], args.learning_rate).to(device)
    model = torch.load(MODEL_PATH).to(device)
    # with open(EXP_PATH + "args" + ".txt", 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    dir_list = os.listdir(DATA_PATH)

    n_traj = 0

    for name in dir_list:
        if name[0:4] == 'traj':
            n_traj += 1
    print(n_traj)
    valid_traj_list = []
    n_data = 0
    for i_traj in range(n_traj):
        name = 'traj_' + str(i_traj).zfill(6)
        dir_name = DATA_PATH + name + '/'
        traj_file = dir_name + 'data.pkl'
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)
        n_frames = len(data['ego'])
        N = n_frames - (args.rollout-1) * args.skip_frame
        if N < 10:
            continue
        valid_traj_list.append(i_traj)
        n_data += N
    print(n_data)

    total_data = []
    count = 0
    for i_traj in range(n_traj):
        name = 'traj_' + str(i_traj).zfill(6)
        dir_name = DATA_PATH + name + '/'
        traj_file = dir_name + 'data.pkl'
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)
        n_frames = len(data['ego'])
        N = n_frames - (args.rollout) * args.skip_frame

        print(i_traj)
        for idx in range(N):
            x = torch.from_numpy(np.array(data['ego'][idx:idx+args.skip_frame*args.rollout:args.skip_frame])).type(torch.FloatTensor)
            
            cluster = cluster_id[count].item()

            heading = []
            for i in range(x.shape[0]):
                if i == x.shape[0] - 1:
                    heading.append(heading[-1])
                else:
                    delta = x[i+1] - x[i]
                    heading.append(math.atan(delta[1].item() / (delta[0].item() + 1e-7)))

            heading = torch.tensor(heading)
            heading = heading.view(-1,1)

            x = torch.cat((x, heading),1)

            action = data['ego'][idx + args.skip_frame * args.rollout] # action = next state

            data_dic = {}
            data_dic["state"] = x
            data_dic["action"] = action
            data_dic["cluster"] = cluster
            data_dic["car_id"] = i_traj
            total_data.append(data_dic)
            count += 1

    with open(DATASET_PATH + "GAIL_data.pkl", 'wb') as f:
        pickle.dump(total_data, f)