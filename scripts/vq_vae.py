from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import argparse


from six.moves import xrange

import umap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from utils import *
# from model import *
from hojun.model import *
from dataloader import *

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
DATASET_PATH = PROJECT_PATH + "/dataset/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--exp_name', default='NGSIM_220620', type=str, help='experiment name')
    parser.add_argument('--data_name', default='NGSIM', type=str, help='dataset name')
    parser.add_argument('--policy_file', default='best', type=str, help='policy file name')
    parser.add_argument('--train', default=True, type=bool, help='training mode')
    parser.add_argument('--rollout', default=20, type=int, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, type=int, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    # parser.add_argument('--n_updates', default=5000, type=int, help='the number of updates in training')
    # parser.add_argument('--n_hiddens', default=128, type=int, help='the number of hidden layers')
    # parser.add_argument('--n_residual_hiddens', default=32, type=int, help='the number of residual hidden layers')
    # parser.add_argument('--n_residual_layers', default=2, type=int, help='the number of residual hidden layers')
    # parser.add_argument('--embedding_dim', default=64, type=int, help='dimension of codebook')
    # parser.add_argument('--n_embedding', default=512, type=int, help='the number of codebook')
    # parser.add_argument('--commitment_cost', default=0.25, type=float, help='commitment cost')
    # parser.add_argument('--decay', default=0.99, type=float, help='')
    # parser.add_argument('--learning_rate', default=1e-3, type=float, help='')


    parser.add_argument('--n_updates', default=200000, help='the number of updates in training')
    parser.add_argument('--num_hiddens', default=(128, 128), help='the number of hidden layers')
    parser.add_argument('--num_residual_hiddens', default=(128, 32), help='the number of residual hidden layers')
    parser.add_argument('--num_residual_layers', default=(2, 2), help='the number of residual hidden layers')
    parser.add_argument('--embedding_dim', default=(1152, 128), help='dimension of codebook')
    parser.add_argument('--num_embeddings', default=(1024, 256), help='the number of codebook')
    parser.add_argument('--commitment_cost', default=0.1, help='commitment cost')
    parser.add_argument('--decay', default=0.99, help='')
    parser.add_argument('--img_shape', default=(117, 24), help='input image resolution')
    parser.add_argument('--dim_ratio', default = 12, help='img size reuction ratio')
    parser.add_argument('--output_padding', default=[(0,0,0,0,1,0), (1,1,0,1)])  ## should be removed
    parser.add_argument('--learning_rate', default=3e-4, help='')
    parser.add_argument('--pretrained_epochs', default=0, help='')
    parser.add_argument('--channel_type', default=[0], help='select rgb')
    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")
    init_time = time.time()

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"

    make_dir(EXP_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])

    # train_data = CarlaDataset(DATA_PATH, 'train', transform, args)
    # valid_data = CarlaDataset(DATA_PATH, 'valid', transform, args)
    # test_data = CarlaDataset(DATA_PATH, 'test', transform, args)

    train_data = NGSIMDataset(DATA_PATH, 'train', transform=transform, args=args)
    valid_data = NGSIMDataset(DATA_PATH, 'valid', transform=transform, args=args)
    test_data = NGSIMDataset(DATA_PATH, 'test', transform=transform, args=args)

    # data_variance = np.var(train_data.data / 255.0)
    data_variance = 1.0

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_data,
                              batch_size=4,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                              batch_size=4,
                              shuffle=True,
                              pin_memory=True)
    

    if args.train:
        # model = Model(args).to(device)
        model = Model_sep(args.num_hiddens, args.num_residual_layers,
                    args.num_residual_hiddens, args.num_embeddings, args.embedding_dim,
                    args.output_padding, args.commitment_cost, args.img_shape, args.dim_ratio, args.channel_type, args.decay).to(device)
        # model = torch.load(POLICY_FILE)
    else:
        model = torch.load(POLICY_FILE)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

    train_res_recon_error = []
    train_res_perplexity = []
    if args.train:
        optimal_recon = None
        for i in range(args.n_updates):
            (data, _) = next(iter(train_loader))
            # print(data.shape)
            data = data.permute(0,2,1,3,4) # hojun part
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 200 == 0:
                print('[%.3f] %d iterations' % (time.time() - init_time, i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            if (i+1) % 5000 == 0:
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                torch.save(model, EXP_PATH + "model_"+str(i+1).zfill(6)+".pt")
                if optimal_recon is None or optimal_recon > np.mean(train_res_recon_error[-100:]):
                    torch.save(model, EXP_PATH + "best.pt")
                    optimal_recon = np.mean(train_res_recon_error[-100:])
                print('[%.3f] %d iterations' % (time.time() - init_time, i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))

            if (i+1) % 500 == 0:
                bev_test = data.permute(0,2,1,3,4)
                bev_test = bev_test.reshape(args.rollout * args.batch_size,len(args.channel_type),args.img_shape[0],args.img_shape[1])
                bev_recon_test = data_recon.permute(0,2,1,3,4)
                bev_recon_test = bev_recon_test.reshape(args.rollout * args.batch_size,len(args.channel_type),args.img_shape[0],args.img_shape[1])

                save_result(make_grid(bev_test[10:30].cpu() + 0.5), EXP_PATH + "original_"+str(i+1).zfill(6)+".png")
                save_result(make_grid(bev_recon_test[10:30].cpu() + 0.5), EXP_PATH + "recon_"+str(i+1).zfill(6)+".png")


    if not args.train:
        # (test_originals, _) = next(iter(test_loader))
        # test_originals = test_originals.to(device)

        # vq_output_eval = model._pre_vq_conv(model._encoder(test_originals))
        # _, test_quantize, _, _, idx = model._vq_vae(vq_output_eval)

        # test_reconstructions = model._decoder(test_quantize)
        # test_shape = test_originals.shape

        # test_originals = torch.reshape(test_originals, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
        # test_reconstructions = torch.reshape(test_reconstructions, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
        # show(make_grid(test_originals.cpu()+0.5))
        # show(make_grid(test_reconstructions.cpu()+0.5))
        bev_test, label = next(iter(test_loader))
        # feature_test = next(iter(test_feature_loader))
        shape = bev_test.shape
        bev_test = bev_test.permute(0,2,1,3,4)
        # feature_test = feature_test.permute(0,3,1,2)
        bev_test = bev_test.to(device)
        # feature_test = feature_test.to(device)

        encoded_bev = model.bev_encoder(bev_test)
        # encoded_feature = model.feature_encoder(feature_test)

        z_bev = model._pre_vq_conv_bev(encoded_bev)
        # z_feature = model._pre_vq_conv_feature(encoded_feature)



        loss_bev, code_bev, _, _ = model._vq_vae_bev(z_bev)
        # _, code_feature, _,_ = model._vq_vae_feature(z_feature)

        bev_recon_test = model.bev_decoder(code_bev)
        recon_error = F.mse_loss(bev_test, bev_recon_test)
        # feature_recon_test = model.feature_decoder(code_feature)

        bev_test = bev_test.permute(0,2,1,3,4)
        bev_test = bev_test.reshape(args.rollout * args.batch_size,len(args.channel_type),args.img_shape[0],args.img_shape[1])
        bev_recon_test = bev_recon_test.permute(0,2,1,3,4)
        bev_recon_test = bev_recon_test.reshape(args.rollout * args.batch_size,len(args.channel_type),args.img_shape[0],args.img_shape[1])
        
        print(loss_bev)
        print(recon_error)
        show(make_grid(bev_test[10:30].cpu() + 0.5))
        show(make_grid(bev_recon_test[10:30].cpu() + 0.5))
