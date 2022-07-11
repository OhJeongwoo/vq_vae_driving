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

    parser.add_argument('--exp_name', default='NGSIM_220620_test', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM', help='dataset name')
    parser.add_argument('--data_type', default='valid', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=8, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=10, help='interval between images in trajectory')

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_PATH = DATA_PATH + args.data_type + "/"

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])

    # dataset = CarlaDataset(DATA_PATH, args.data_type, transform, args)
    dataset = NGSIMDataset(DATA_PATH, args.data_type, transform=transform, args=args)
    
    data_variance = 1.0

    data_loader = DataLoader(dataset, 
                             batch_size=1, 
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    model = torch.load(POLICY_FILE)

    N = len(dataset)
    print(N)
    n_data = 0
    for batch in data_loader:
        input = batch[0]
        i_traj = batch[1]['traj'].cpu().item()
        seq = batch[1]['seq'].cpu().item()
        # print("traj #: %d, seq #: %d" %(i_traj, seq))
        traj_name = 'traj_' + str(i_traj).zfill(6)
        seq_name = str(seq).zfill(6)
        file_name = SAVE_PATH + traj_name + "/" + seq_name + ".pkl"
        '''
        original version
        '''
        # data_originals = input.to(device)
        # vq_output_eval = model._pre_vq_conv(model._encoder(data_originals))
        # _, _, _, _, embbeding_idx = model._vq_vae(vq_output_eval)
        # embbeding_idx = np.squeeze(embbeding_idx.cpu().numpy(), axis=0)
        
        '''
        hojun version
        '''
        bev_test = input.permute(0,2,1,3,4)
        bev_test = bev_test.to(device)
        encoded_bev = model.bev_encoder(bev_test)
        z_bev = model._pre_vq_conv_bev(encoded_bev)
        loss_bev, code_bev, _, embedding_idx = model._vq_vae_bev(z_bev)
        embedding_idx = torch.squeeze(embedding_idx.permute(1,0,2,3),dim=0)
        # you need to extract embbeding index from quantizer code!
        
        with open(file_name, 'rb') as f:
            label = pickle.load(f)
        label['code'] = embedding_idx.cpu().numpy()
        
        with open(file_name, 'wb') as f:
            pickle.dump(label, f)
        f.close()
        n_data += 1
        if n_data % 1000 == 0:
            print("[%.3f] %d/%d, expected remaining time: %.3f" %(time.time() - init_time, n_data, N, (time.time() - init_time) / n_data * (N - n_data)))



