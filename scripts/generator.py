from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse


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
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--train', default=True, help='training mode')
    parser.add_argument('--rollout', default=10, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='batch size')
    parser.add_argument('--n_updates', default=30000, help='the number of updates in training')
    parser.add_argument('--n_hiddens', default=128, help='the number of hidden layers')
    parser.add_argument('--n_residual_hiddens', default=32, help='the number of residual hidden layers')
    parser.add_argument('--n_residual_layers', default=2, help='the number of residual hidden layers')
    parser.add_argument('--embedding_dim', default=64, help='dimension of codebook')
    parser.add_argument('--n_embedding', default=512, help='the number of codebook')
    parser.add_argument('--commitment_cost', default=0.25, help='commitment cost')
    parser.add_argument('--decay', default=0.99, help='')
    parser.add_argument('--learning_rate', default=1e-3, help='')

    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])

    # train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    # valid_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_data = CarlaDataset(DATA_PATH, 'train', transform, args)
    valid_data = CarlaDataset(DATA_PATH, 'valid', transform, args)
    test_data = CarlaDataset(DATA_PATH, 'test', transform, args)

    # data_variance = np.var(train_data.data / 255.0)
    data_variance = 1.0

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=32,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                              batch_size=16,
                              shuffle=True,
                              pin_memory=True)



    if args.train:
        model = Model(args).to(device)
    else:
        model = torch.load(POLICY_FILE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)



    model.train()
    train_res_recon_error = []
    train_res_perplexity = []


    # generate index


    test_quantize = model._vq_vae.load_codebook(idx)

    test_reconstructions = model._decoder(test_quantize)
    
    test_reconstructions = torch.reshape(test_originals, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
    show(make_grid(test_reconstructions.cpu()+0.5))
        
    model.eval()