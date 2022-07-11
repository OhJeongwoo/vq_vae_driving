
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse
import pickle
import time
import random
from sklearn import cluster

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
DATASET_PATH = PROJECT_PATH + "/dataset/"
DATA_PATH = DATASET_PATH + "latent.pkl"
CARID_PATH = DATASET_PATH + "car_id.pkl"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--n_cluster', default=10, type =int, help='# of cluster')
    parser.add_argument('--convergence_eps', default=0.99999, type=float, help='stopping criterion')

    args = parser.parse_args()

    with open(DATA_PATH, 'rb') as f:
        latent_data = pickle.load(f)
    with open(CARID_PATH, 'rb') as f:
        car_id = pickle.load(f)

    n_cluster = args.n_cluster
    len_data = latent_data.shape[0]
    dimension = latent_data.shape[2]

    initial_center_index = random.sample(range(0, len_data),10)
    centers = latent_data[initial_center_index]
    centers = centers.view(1, n_cluster, dimension)

    cluster_num = torch.zeros(len_data, dtype=int)
    eps = 0.001

    while(eps < args.convergence_eps):
        
        old_centers = torch.tensor(centers)

        distances = F.cosine_similarity(latent_data, centers, dim=-1)
        # print(distances.shape)

        cluster_num = torch.argmax(distances, dim=1)
        # print(cluster_num)

        for i_cluster in range(n_cluster):
            centers[:, i_cluster, :] = torch.mean(latent_data[cluster_num[:] == i_cluster], dim=0)
        # print(centers[:,0,:])
        # print(old_centers[:,0,:])

        eps = torch.min(F.cosine_similarity(centers, old_centers, dim=-1))
    
    with open(DATASET_PATH + "cluster_id.pkl", 'wb') as f:
        pickle.dump(cluster_num, f)

    with open(DATASET_PATH + "kmeans_centers.pkl", 'wb') as f:
        pickle.dump(centers, f)
