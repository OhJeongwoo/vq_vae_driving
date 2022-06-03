import os
import math
import numpy as np
from dataloader import CarlaDataset, CodeDataset
import argparse

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')


from matplotlib.colors import to_rgb

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary

    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from model import *
from utils import *

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/Carla_high"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

import urllib.request
from urllib.error import HTTPError

# Convert images from 0-1 to 0-255 (integers). We use the long datatype as we will use the images as labels as well


# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(),
                                discretize])    
DATA_PATH = "../dataset/Carla_high"



# We define a set of data loaders that we can use for various purposes later.
# TODO : Latent space dataloader
train_set = CodeDataset(DATA_PATH, 'train', transform)
val_set = CodeDataset(DATA_PATH, 'valid', transform)
test_set = CodeDataset(DATA_PATH, 'test', transform)


train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=1)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=1)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=1)



# def show_imgs(imgs):
#     num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
#     nrow = min(num_imgs, 4)
#     ncol = int(math.ceil(num_imgs/nrow))
#     imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128)
#     imgs = imgs.clamp(min=0, max=255)
#     np_imgs = imgs.cpu().numpy()
#     plt.figure(figsize=(1.5*nrow, 1.5*ncol))
#     plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
#     plt.axis('off')
#     plt.show()
#     plt.close()

#show_imgs([train_set[i][0] for i in range(8)])


def train_model(**kwargs):
    # trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "PixelCNN"),
    #                      gpus=1 if str(device).startswith("cuda") else 0,
    #                      max_epochs=10,
    #                      callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
    #                                 LearningRateMonitor("epoch")])

    # model = PixelCNN(**kwargs)
    # trainer.fit(model, train_loader, val_loader)
    # model = model.to(device)

    

    model = PixelCNN(**kwargs)

    for epoch in range(1000):
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "PixelCNN"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")])
        trainer.fit(model, train_loader, val_loader)

        if epoch % 10 == 0:
            torch.save(model.to(device), CHECKPOINT_PATH + "/model_" + str(epoch).zfill(5) + ".pt")
            print("save model")
    model = model.to(device)
    
    return model

model = train_model(c_in=1, c_hidden=64)
torch.save(model, "pixel_final.pt")



# test_res = result["test"][0]
# print("Test bits per dimension: %4.3fbpd" % (test_res["test_loss"] if "test_loss" in test_res else test_res["test_bpd"]))

# num_params = sum([np.prod(param.shape) for param in model.parameters()])
# print("Number of parameters: {:,}".format(num_params))

#sampling
pl.seed_everything(1)
samples = model.sample(img_shape=(16,1,17,30))
# show_imgs(samples.cpu())