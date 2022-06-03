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
import pytorch_lightning as pl

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

    parser.add_argument('--exp_name', default='Carla_220417', help='experiment name')
    #parser.add_argument('--data_name', default='Carla_high', help='dataset name')
    parser.add_argument('--data_name', default='Carla', help='dataset name')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=5, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=5, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=64, help='batch size')
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



    model = torch.load(POLICY_FILE)
    # generator = PixelCNN.load_from_checkpoint(PROJECT_PATH + "/saved_models/tutorial12_dupl/PixelCNN.ckpt")
    # generator = torch.load(PROJECT_PATH+"/saved_models/Carla_high/model_00990.pt")
    generator = torch.load(PROJECT_PATH+"/saved_models/220417/model_00490.pt")
    # generator = torch.load(PROJECT_PATH + "/saved_models/tutorial12_dupl/PixelCNN.ckpt")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)



    model.train()
    train_res_recon_error = []
    train_res_perplexity = []


    # generate index

    pl.seed_everything(2)
    # samples = generator.sample(img_shape=(8,1,33,60))
    # samples = generator.sample(img_shape = (10, 1, 17, 30))
    # print(samples[1])
    # print(samples.shape)
    # print(samples)
    # samples = int(samples)
    # print(samples)
    test_quantize = model._vq_vae.load_codebook(samples)

    test_reconstructions = model._decoder(test_quantize)
    test_shape = test_reconstructions.shape
    test_reconstructions = torch.reshape(test_reconstructions, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
    print(test_reconstructions.shape)
    print(test_reconstructions[0:10])
    show(make_grid(test_reconstructions.cpu()+0.5))
        
    model.eval()