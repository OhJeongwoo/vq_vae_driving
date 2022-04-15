from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
import pickle


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
from dataloader import CarlaDataset

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
DATASET_PATH = PROJECT_PATH + "/dataset/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--exp_name', default='Carla', help='experiment name')
    parser.add_argument('--data_name', default='Carla', help='dataset name')
    parser.add_argument('--data_type', default='train', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=10, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_FILE = DATA_PATH + args.data_type + '/label.pkl'

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])

    dataset = CarlaDataset(DATA_PATH, args.data_type, transform, args)

    # data_variance = np.var(train_data.data / 255.0)
    data_variance = 1.0

    data_loader = DataLoader(dataset, 
                             batch_size=1, 
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    model = torch.load(POLICY_FILE)

    for batch in data_loader:
        input = batch[0]
        i_traj = batch[1]['traj'].cpu().item()
        seq = batch[1]['seq'].cpu().item()
        print("traj #: %d, seq #: %d" %(i_traj, seq))
        traj_name = 'traj_' + str(i_traj).zfill(6)
        seq_name = 'seq_' + str(seq).zfill(6)
        data_originals = input.to(device)
        vq_output_eval = model._pre_vq_conv(model._encoder(data_originals))
        _, _, _, _, embbeding_idx = model._vq_vae(vq_output_eval)
        embbeding_idx = np.squeeze(embbeding_idx.cpu().numpy(), axis=0)
        dataset.label[traj_name][seq_name]['label'] = embbeding_idx
    
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset.label, f)



