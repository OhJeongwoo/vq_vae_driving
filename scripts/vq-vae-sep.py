from __future__ import print_function

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from utils.dataset import ngsim_feature_dataset, ngsim_bev_dataset
from hojun.model import Model_sep

def show(img):
    npimg = img.detach().numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data processor for NGSIM dataset')

    # parser.add_argument('--exp_name', default='Carla', help='experiment name')
    # parser.add_argument('--data_name', default='Carla', help='dataset name')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--mode', default='train', help='training mode')
    # parser.add_argument('--rollout', default=10, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default= 16, help='batch size')
    parser.add_argument('--in_channels', default=(3, 4), help='batch size')
    parser.add_argument('--n_updates', default=20000, help='the number of updates in training')
    parser.add_argument('--num_hiddens', default=(1024, 128), help='the number of hidden layers')
    parser.add_argument('--num_residual_hiddens', default=(256, 32), help='the number of residual hidden layers')
    parser.add_argument('--num_residual_layers', default=(2, 2), help='the number of residual hidden layers')
    parser.add_argument('--embedding_dim', default=(2048, 256), help='dimension of codebook')
    parser.add_argument('--num_embeddings', default=(3072, 256), help='the number of codebook')
    parser.add_argument('--commitment_cost', default=0.1, help='commitment cost')
    parser.add_argument('--decay', default=0.99, help='')
    parser.add_argument('--img_shape', default=(117, 24), help='input image resolution')
    parser.add_argument('--output_padding', default=[(1,0,0,0,1,0), (1,1,0,1)])
    parser.add_argument('--learning_rate', default=4e-5, help='')
    parser.add_argument('--pretrained_epochs', default=0, help='')
    args = parser.parse_args()
    
    BEV_PATH = "data/bev/"
    FEATURE_PATH = "data/features/"
    MODEL_PATH = "log_sep/models/best.pt"
    EXP_PATH = "log_sep/models/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_bev = ngsim_bev_dataset(path=BEV_PATH, mode = "train")
    train_feature = ngsim_feature_dataset(path=FEATURE_PATH, mode = "train")

    val_bev = ngsim_bev_dataset(path=BEV_PATH, mode = "val")
    val_feature = ngsim_feature_dataset(path=FEATURE_PATH, mode = "val")

    test_bev = ngsim_bev_dataset(path=BEV_PATH, mode = "test")
    test_feature = ngsim_feature_dataset(path=FEATURE_PATH, mode = "test")

    # bev_variance = np.var(train_bev)
    # feature_variance = np.var(train_feature)

    # bev_variance_val = np.var(val_bev)
    # feature_variance_val = np.var(val_feature)

    train_bev_loader = DataLoader(train_bev, 
                             batch_size=args.batch_size, 
                             shuffle=False,
                             pin_memory=True)

    train_feature_loader = DataLoader(train_feature, 
                             batch_size=args.batch_size, 
                             shuffle=False,
                             pin_memory=True)
    
    val_bev_loader = DataLoader(val_bev, 
                             batch_size=8, 
                             shuffle=False,
                             pin_memory=True)
    
    val_feature_loader = DataLoader(val_feature, 
                             batch_size=8, 
                             shuffle=False,
                             pin_memory=True)

    test_bev_loader = DataLoader(test_bev, 
                             batch_size=4, 
                             shuffle=False,
                             pin_memory=True)
    
    test_feature_loader = DataLoader(test_feature, 
                             batch_size=4, 
                             shuffle=False,
                             pin_memory=True)
    
    
    if args.mode == 'train':
        model = Model_sep(args.in_channels, args.num_hiddens, args.num_residual_layers,
                    args.num_residual_hiddens, args.num_embeddings, args.embedding_dim,
                    args.output_padding, args.commitment_cost, args.decay).to(device)
        # model = torch.load(MODEL_PATH)
    else:
        model = torch.load(MODEL_PATH)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)
    schedular = StepLR(optimizer, step_size=4000, gamma=0.7)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    val_recon_error = []
    
    if args.mode ==  "train":
        optimal_recon = None
        print("lets start")
        for i in xrange(args.n_updates):

            bev = next(iter(train_bev_loader))
            feature = next(iter(train_feature_loader))
            bev = bev.permute(0,2,1,3,4)
            feature = feature.permute(0,3,1,2)

            bev = bev.contiguous()
            feature = feature.contiguous()

            bev = bev.to(device)
            feature = feature.to(device)
            model.train()
            optimizer.zero_grad()

            vq_loss, bev_recon, feature_recon, perplexity = model.forward(bev, feature)
            bev_recon_error = F.mse_loss(bev_recon, bev) / torch.var(bev)
            feature_recon_error = F.mse_loss(feature_recon, feature) / torch.var(feature)

            recon_error = bev_recon_error + feature_recon_error
            loss = recon_error + vq_loss

            loss.backward()

            optimizer.step()
            schedular.step()
            
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                torch.save(model, EXP_PATH + "model_"+str(i+1).zfill(3) + ".pt")
                if optimal_recon is None or optimal_recon > np.mean(train_res_recon_error[-100:]):
                    torch.save(model, EXP_PATH + "best.pt")
                    optimal_recon = np.mean(train_res_recon_error[-100:])
                

                f = plt.figure(figsize=(16,8))
                ax = f.add_subplot(1,2,1)
                ax.plot(train_res_recon_error, label='training')
                # ax.plot(val_recon_error_smooth, label='validation')
                ax.legend
                ax.set_yscale('log')
                ax.set_title('Smoothed NMSE.')
                ax.set_xlabel('iteration')

                ax = f.add_subplot(1,2,2)
                ax.plot(train_res_perplexity)
                ax.set_title('Smoothed Average codebook usage (perplexity).')
                ax.set_xlabel('iteration')

                plt.savefig("log_sep/training_logs/" +str(i+1).zfill(3) + ".png")
            # plt.show()
            # model.eval()
            # bev_val = next(iter(val_bev_loader))
            # feature_val = next(iter(val_feature_loader))

            # bev_val = bev_val.permute(0,2,1,3,4)
            # feature_val = feature_val.permute(0,3,1,2)

            # bev_val = bev_val.to(device)
            # feature_val = feature_val.to(device)

            # vq_loss_val, bev_recon_val, feature_recon_val, perplexity_val = model.forward(bev_val, feature_val)
            # bev_recon_error_val = F.mse_loss(bev_recon_val, bev_val) 
            # feature_recon_error_val = F.mse_loss(feature_recon_val, feature_val) 
            # recon_error_val = bev_recon_error_val + feature_recon_error_val
            # val_recon_error.append(recon_error_val.item())
            # #TODO: EVAL CODE

            # train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
            # train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
            # val_recon_error_smooth = savgol_filter(val_recon_error, 201, 7)
            



    if not args.mode == "train":
        
        bev_test = next(iter(test_bev_loader))
        feature_test = next(iter(test_feature_loader))

        bev_test = bev_test.permute(0,2,1,3,4)
        feature_test = feature_test.permute(0,3,1,2)
        bev_test = bev_test.to(device)
        feature_test = feature_test.to(device)

        encoded_bev = model.bev_encoder(bev_test)
        encoded_feature = model.feature_encoder(feature_test)

        z_bev = model._pre_vq_conv_bev(encoded_bev)
        z_feature = model._pre_vq_conv_feature(encoded_feature)



        loss_bev, code_bev, _, _ = model._vq_vae_bev(z_bev)
        _, code_feature, _,_ = model._vq_vae_feature(z_feature)

        bev_recon_test = model.bev_decoder(code_bev)
        feature_recon_test = model.feature_decoder(code_feature)

        bev_test = bev_test.permute(0,2,1,3,4)
        bev_test = bev_test.reshape(120,3,117,24)
        bev_recon_test = bev_recon_test.permute(0,2,1,3,4)
        bev_recon_test = bev_recon_test.reshape(120,3,117,24)

        print(loss_bev)
        show(make_grid(bev_test[10:30].cpu()))
        show(make_grid(bev_recon_test[10:30].cpu()))
        # test_originals = torch.reshape(test_originals, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
        # test_reconstructions = torch.reshape(test_originals, (test_shape[0] * test_shape[1] // 3, 3, test_shape[2], test_shape[3]))
        # show(make_grid(test_originals.cpu()+0.5))
        # show(make_grid(test_reconstructions.cpu()+0.5))
        
    