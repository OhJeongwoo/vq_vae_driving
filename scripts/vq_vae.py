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

    parser.add_argument('--exp_name', default='cifar', help='experiment name')
    parser.add_argument('--data_name', default='Carla', help='dataset name')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--train', default=False, help='training mode')
    parser.add_argument('--rollout', default=1, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='batch size')
    parser.add_argument('--n_updates', default=15000, help='the number of updates in training')
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

    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # train_data = CarlaDataset(DATA_PATH, 'train', transform, args)
    # valid_data = CarlaDataset(DATA_PATH, 'valid', transform, args)
    # test_data = CarlaDataset(DATA_PATH, 'test', transform, args)

    data_variance = np.var(train_data.data / 255.0)

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=32,
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

    if args.train:
        optimal_recon = None
        for i in xrange(args.n_updates):
            (data, _) = next(iter(train_loader))
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()
            
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                torch.save(model, EXP_PATH + "model_"+str(i+1).zfill(3)+".pt")
                if optimal_recon is None or optimal_recon > np.mean(train_res_recon_error[-100:]):
                    torch.save(model, EXP_PATH + "best.pt")
                    optimal_recon = np.mean(train_res_recon_error[-100:])
                print()

        train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)


        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')

        ax = f.add_subplot(1,2,2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')

        plt.show()


    model.eval()

    (valid_originals, _) = next(iter(valid_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)


    (train_originals, _) = next(iter(train_loader))
    train_originals = train_originals.to(device)
    _, train_reconstructions, _, _ = model._vq_vae(train_originals)


    def show(img):
        npimg = img.numpy()
        fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()


    show(make_grid(valid_reconstructions.cpu().data)+0.5, )


    show(make_grid(valid_originals.cpu()+0.5))


    proj = umap.UMAP(n_neighbors=3,
                    min_dist=0.1,
                    metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())

    plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
    plt.show()