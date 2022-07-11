from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
import pickle
import time


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

    parser.add_argument('--exp_name', default='NGSIM_feat_220625', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM_feature', help='dataset name')
    parser.add_argument('--data_type', default='train', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=128, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/" + args.data_type + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_PATH = DATA_PATH + args.data_type + "/"

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(POLICY_FILE)
    dir_list = os.listdir(DATA_PATH)
    n_traj = 0
    for name in dir_list:
        if name[0:4] == 'traj':
            n_traj += 1
    print(n_traj)
    for i_traj in range(n_traj):
        print(i_traj)
        name = 'traj_' + str(i_traj).zfill(6)
        dir_name = DATA_PATH + name + '/'
        traj_file = dir_name + 'data.pkl'
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)
        n_frames = len(data['ego'])
        N = n_frames - (args.rollout-1) * args.skip_frame
        codes = []
        for seq in range(N):
            x = torch.from_numpy(np.array(data['ego'][seq:seq+args.skip_frame*args.rollout:args.skip_frame])).type(torch.FloatTensor)
            x = torch.unsqueeze(torch.unsqueeze(x, dim=0),dim=0).to(device)
            encoded_bev = model.feature_encoder(x)
            z_bev = model._pre_vq_conv(encoded_bev)
            loss_bev, code_bev, _, embedding_idx = model._vq_vae(z_bev)
            print(embedding_idx)
            codes.append(embedding_idx.cpu().numpy())
        # data['code'] = np.array(codes)
        # print(data['code'].shape)
        # pk = open(traj_file, 'wb')
        # pickle.dump(data, pk)

    # N = len(dataset)
    # print(N)
    # n_data = 0
    # for batch in data_loader:
    #     input = batch[0]
    #     i_traj = batch[1]['traj'].cpu().item()
    #     seq = batch[1]['seq'].cpu().item()
    #     # print("traj #: %d, seq #: %d" %(i_traj, seq))
    #     traj_name = 'traj_' + str(i_traj).zfill(6)
    #     seq_name = str(seq).zfill(6)
    #     file_name = SAVE_PATH + traj_name + "/" + seq_name + ".pkl"
    #     '''
    #     original version
    #     '''
    #     # data_originals = input.to(device)
    #     # vq_output_eval = model._pre_vq_conv(model._encoder(data_originals))
    #     # _, _, _, _, embbeding_idx = model._vq_vae(vq_output_eval)
    #     # embbeding_idx = np.squeeze(embbeding_idx.cpu().numpy(), axis=0)
        
    #     '''
    #     hojun version
    #     '''
    #     bev_test = input.permute(0,2,1,3,4)
    #     bev_test = bev_test.to(device)
    #     encoded_bev = model.bev_encoder(bev_test)
    #     z_bev = model._pre_vq_conv_bev(encoded_bev)
    #     loss_bev, code_bev, _, embedding_idx = model._vq_vae_bev(z_bev)
    #     embedding_idx = torch.squeeze(embedding_idx.permute(1,0,2,3),dim=0)
    #     # you need to extract embbeding index from quantizer code!
        
    #     with open(file_name, 'rb') as f:
    #         label = pickle.load(f)
    #     label['code'] = embedding_idx.cpu().numpy()
        
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(label, f)
    #     f.close()
    #     n_data += 1
    #     if n_data % 1000 == 0:
    #         print("[%.3f] %d/%d, expected remaining time: %.3f" %(time.time() - init_time, n_data, N, (time.time() - init_time) / n_data * (N - n_data)))



