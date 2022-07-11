from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
import pickle
import time
import random


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
from rl_model import *
from dataloader import *

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
DATASET_PATH = PROJECT_PATH + "/dataset/"

if __name__ == '__main__':
    init_time = time.time() 

    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--exp_name', default='dynamics_6', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM_feature', help='dataset name')
    parser.add_argument('--data_type', default='train', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=10, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--feature_dim', default=5, help='interval between images in trajectory')
    parser.add_argument('--learning_rate', default=5e-6, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/" + args.data_type + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_PATH = DATA_PATH + args.data_type + "/"
    MODEL_PATH = EXP_PATH + "best.pt"

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(args.rollout * args.feature_dim + 2, 2, [64, 32], args.learning_rate, device, [None, 'leaky-relu', None]).to(device)
    with open(EXP_PATH + "args" + ".txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
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
        N = n_frames - args.rollout * args.skip_frame
        if N < 10:
            continue
        valid_traj_list.append(i_traj)
        n_data += N
    print(n_data)

    best_loss = None
    iter_list = []
    loss_all = []
    pos_all = []
    neg_all = []
    losses = []
    pos_sims = []
    neg_sims = []
    for i_iter in range(1000000):
        x_list = []
        y_list = []
        for i in range(args.batch_size):
            i_traj = random.choice(valid_traj_list)
            name = 'traj_' + str(i_traj).zfill(6)
            dir_name = DATA_PATH + name + '/'
            traj_file = dir_name + 'data.pkl'
            with open(traj_file, 'rb') as f:
                neg_data = pickle.load(f)
            n_frames = len(neg_data['ego'])
            N = n_frames - args.rollout * args.skip_frame
            idx = random.choice(range(0,N))
            z = torch.from_numpy(np.array(neg_data['ego'][idx:idx+args.skip_frame*(args.rollout+1):args.skip_frame])).type(torch.FloatTensor)
            z = torch.flatten(z).to(device)
            # x = torch.cat((z[:args.rollout * args.feature_dim], z[args.rollout * args.feature_dim + 3:]))
            # y = z[args.rollout * args.feature_dim: args.rollout * args.feature_dim + 3]
            # x = z[:args.rollout * args.feature_dim+3]
            # y = z[args.rollout * args.feature_dim + 3:]
            x = torch.cat((z[:args.rollout * args.feature_dim], z[args.rollout * args.feature_dim + 3:]))
            y = z[args.rollout * args.feature_dim: args.rollout * args.feature_dim + 2]
            
            x_list.append(x)
            y_list.append(y)
        x = torch.stack(x_list, dim=0)
        y = torch.stack(y_list, dim=0)
        predict_y = model(x)
        
        loss = F.mse_loss(y, predict_y)
        loss.backward()
        model.optimizer.step()
        # z_pos = model(x_pos)
        # z_neg = model(x_neg)

        # pos_cos_sim = F.cosine_similarity(z,z_pos)
        # neg_cos_sim = F.cosine_similarity(z,z_neg)
        # pos_val = torch.exp(pos_cos_sim / args.temperature)
        # neg_val = torch.sum(torch.exp(neg_cos_sim / args.temperature))
        # loss = -torch.log(pos_val / (pos_val + neg_val))
        # loss.backward()
        # model.optimizer.step()

        losses.append(loss)
        # pos_sims.append(pos_cos_sim)
        # neg_sims.append(torch.mean(neg_cos_sim))


        if i_iter > 0 and i_iter % 100 == 0:
            # print(y)
            # print(predict_y)
            print(y[:5])
            print(y[:5] - predict_y[:5])
            cur_loss = torch.mean(torch.stack(losses)).item()
            iter_list.append(i_iter)
            loss_all.append(cur_loss)
            print("[%.3f] iter: %d, loss: %.3f" %(time.time() - init_time, i_iter, cur_loss))
            if best_loss is None or cur_loss < best_loss:
                torch.save(model, EXP_PATH + "best.pt")
                best_loss = cur_loss
            plt.plot(iter_list, loss_all, color='black')
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.savefig(EXP_PATH + 'result.png')
            plt.clf()
            
            losses = []