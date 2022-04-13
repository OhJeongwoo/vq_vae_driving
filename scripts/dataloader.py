import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image

from utils import *

class CarlaDataset(Dataset):
    def __init__(self, path, mode, transform=None, args=None):
        # set hyperparamter
        self.path = path
        self.mode = mode
        self.transform = transform
        if args is None:
            self.rollout = 10
            self.skip_frame = 1
        else:
            self.rollout = args.rollout
            self.skip_frame = args.skip_frame
        
        if self.mode == 'train':
            self.path += '/train/'
        elif self.mode == 'valid':
            self.path += '/valid/'
        elif self.mode == 'test':
            self.path += '/test/'
        else:
            print("Invalid mode %s" %(self.mode))
            return
        
        dir_list = os.listdir(self.path)
        self.n_traj = 0
        for name in dir_list:
            if name[0:4] == 'traj':
                self.n_traj += 1

        self.data_list = []
        for i_traj in range(self.n_traj):
            dir_name = self.path + 'traj_' + str(i_traj).zfill(4) + '/'
            n_frames = count_files(dir_name)
            for seq in range(n_frames - (self.rollout - 1) * self.skip_frame):
                self.data_list.append({'traj': i_traj, 'seq': seq})


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        i_traj = self.data_list[idx]['traj']
        seq = self.data_list[idx]['seq']
        img_path = self.path + 'traj_' + str(i_traj).zfill(4) + '/'
        imgs = []
        for k in range(self.rollout):
            img_file = img_path + str(seq + k * self.skip_frame).zfill(6) + '.png'
            img = Image.open(img_file)
            if self.transform is not None:
                imgs.append(self.transform(img))
        
        return torch.cat(imgs, dim=0), None

    def get_variance(self):
        # calculate variance of dataset
        # output C*H*W
        return 1.0
    

if __name__=="__main__":
    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])
    
    dataset = CarlaDataset(t=5)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            shuffle=True,
                            drop_last=True)

    for epoch in range(2):
        print("***")
        for batch in dataloader:
            print(batch.shape)