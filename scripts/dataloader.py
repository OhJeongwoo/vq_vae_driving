import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
import pickle
import argparse

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
        
        self.label_file = self.path + 'label.pkl'
        self.label_exists = True
        if os.path.exists(self.label_file):
            with open(self.label_file, 'rb') as pf:
                self.label = pickle.load(pf)
        else:
            self.label = {'rollout': self.rollout, 'skip_frame': self.skip_frame}
            self.label_exists = False

        dir_list = os.listdir(self.path)
        self.n_traj = 0
        for name in dir_list:
            if name[0:4] == 'traj':
                self.n_traj += 1

        self.data_list = []
        for i_traj in range(self.n_traj):
            name = 'traj_' + str(i_traj).zfill(6)
            dir_name = self.path + name + '/'
            n_frames = count_files(dir_name, cond='png')
            if not self.label_exists:
                    self.label[name] = {}
            for seq in range(n_frames - (self.rollout - 1) * self.skip_frame):
                self.data_list.append({'traj': i_traj, 'seq': seq})
                if not self.label_exists:
                    self.label[name]['seq_' + str(seq).zfill(6)] = {'traj': i_traj, 'seq': seq}


    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        i_traj = self.data_list[idx]['traj']
        seq = self.data_list[idx]['seq']
        traj_name = 'traj_' + str(i_traj).zfill(6)
        seq_name = 'seq_' + str(seq).zfill(6)
        img_path = self.path + traj_name + '/'
        imgs = []
        for k in range(self.rollout):
            img_file = img_path + str(seq + k * self.skip_frame).zfill(6) + '.png'
            img = Image.open(img_file)
            h, w = img.size
            img = img.crop((0,0,(h//4)*4,(w//4)*4))
            if self.transform is not None:
                imgs.append(self.transform(img))
        
        return torch.cat(imgs, dim=0), self.label[traj_name][seq_name]

    def get_variance(self):
        # calculate variance of dataset
        # output C*H*W
        return 1.0


class NGSIMDataset(Dataset):
    def __init__(self, path, mode, data_format='image', prediction=False, transform=None, args=None):
        # set hyperparamter
        self.path = path
        self.mode = mode
        self.transform = transform
        self.data_format = data_format
        self.prediction = prediction
        if self.prediction:
            self.n_dec_seq = args.n_dec_seq
            self.jump_frame = args.jump_frame
        # self.channels = args.channel_type

        if args is None:
            self.rollout = 8
            self.skip_frame = 10
            self.n_dec_seq = 10
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
        
        self.label_file = self.path + 'label.pkl'
        self.label_exists = True

        dir_list = os.listdir(self.path)
        self.n_traj = 0
        for name in dir_list:
            if name[0:4] == 'traj':
                self.n_traj += 1

        
        self.data_list = []
        if self.data_format == 'feature_code':
            for i_traj in range(self.n_traj):
                name = 'traj_' + str(i_traj).zfill(6)
                dir_name = self.path + name + '/'
                traj_file = dir_name + 'data.pkl'
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                n_frames = len(data['ego'])
                if self.prediction:
                    N = n_frames - self.rollout * self.skip_frame - self.n_dec_seq * self.jump_frame
                else:
                    N = n_frames - (self.rollout - 1) * self.skip_frame
                for seq in range(N):
                    self.data_list.append({'traj': i_traj, 'seq': seq})
        elif self.data_format != 'feature':
            for i_traj in range(self.n_traj):
                name = 'traj_' + str(i_traj).zfill(6)
                dir_name = self.path + name + '/'
                n_frames = count_files(dir_name, cond='pkl')
                if self.prediction:
                    N = n_frames - self.rollout * self.skip_frame - self.n_dec_seq * self.jump_frame
                else:
                    N = n_frames - (self.rollout - 1) * self.skip_frame
                for seq in range(N):
                    self.data_list.append({'traj': i_traj, 'seq': seq})
        else:
            for i_traj in range(self.n_traj):
                name = 'traj_' + str(i_traj).zfill(6)
                dir_name = self.path + name + '/'
                traj_file = dir_name + 'data.pkl'
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                n_frames = len(data['ego'])
                if self.prediction:
                    N = n_frames - self.rollout * self.skip_frame - self.n_dec_seq * self.jump_frame
                else:
                    N = n_frames - (self.rollout - 1) * self.skip_frame
                for seq in range(N):
                    self.data_list.append({'traj': i_traj, 'seq': seq})

        # test_code = []
        # for i in range(10):
        #     test_code.append(self.load_code(0,5*i))
        # for i in range(9):
        #     print(torch.sum(test_code[i] - test_code[i+1] == 0))

    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        i_traj = self.data_list[idx]['traj']
        seq = self.data_list[idx]['seq']       
        
        if self.data_format == 'image':
            # data
            data_imgs = []
            for i in range(self.rollout):
                data_imgs.append(self.load_image(i_traj, seq + i * self.skip_frame))
            # data = torch.cat(data_imgs, dim=0)
            data = torch.stack(data_imgs, dim=0)
            if self.prediction:
                label_imgs = []
                for i in range(1, self.rollout + 1):
                    label_imgs.append(self.load_image(i_traj, seq + i * self.skip_frame))
                label = torch.cat(label_imgs, dim=0)
            else:
                label = 0
        elif self.data_format == 'code':
            data_codes = []
            # for i in range(self.rollout):
            #     data_codes.append(self.load_code(i_traj, seq + i * self.skip_frame))
            # data = torch.cat(data_codes, dim=0)
            # data = self.load_code(i_traj, seq)
            data = []
            for i in range(self.n_dec_seq):
                # code = self.load_code(i_traj, seq + i * self.skip_frame)
                code = self.load_code(i_traj, seq + i * self.jump_frame)
                # code = self.load_code(i_traj, seq + i)
                data.append(code)
            data = torch.cat(data, dim=0)
            if self.prediction:
                # label = self.load_code(i_traj, seq+1)
                # label = self.load_code(i_traj, seq + self.n_dec_seq * self.skip_frame)
                label = self.load_code(i_traj, seq + self.n_dec_seq * self.jump_frame)
                # label = self.load_code(i_traj, seq + self.n_dec_seq)
                label = torch.squeeze(label, dim=0)
                # label_codes = []
                # for i in range(1, self.rollout + 1):
                #     label_codes.append(self.load_code(i_traj, seq + i * self.skip_frame))
                # label = torch.cat(label_codes, dim=0)
            else:
                label = 0
        elif self.data_format == 'feature':
            data = []
            data = self.load_feature(i_traj, seq, self.rollout, self.skip_frame)
            
            if self.prediction:
                label = self.load_feature(i_traj, seq + self.n_dec_seq * self.jump_frame, 1, self.jump_frame)
                label = torch.squeeze(label, dim=0)
            else:
                label = 0
        elif self.data_format == 'feature_code':
            data = []
            data = self.load_feature_code(i_traj, seq, self.n_dec_seq, self.jump_frame)

            if self.prediction:
                label = self.load_feature_code(i_traj, seq + self.n_dec_seq * self.jump_frame, 1, self.jump_frame)
                label = torch.squeeze(label, dim=0)
            else:
                label = 0

        return data, label

    
    def load_image(self, i_traj, seq):
        traj_name = 'traj_' + str(i_traj).zfill(6)
        img_name = str(seq).zfill(6) + '.png'
        img_file = self.path + traj_name + '/' + img_name
        img = Image.open(img_file)
        # h, w = img.size
        # img = img.crop((0,0,(h//4)*4,(w//4)*4))
        # return self.transform(img)[self.channels]
        return self.transform(img)

    def load_code(self, i_traj, seq):
        traj_name = 'traj_' + str(i_traj).zfill(6)
        code_name = str(seq).zfill(6) + '.pkl'
        code_file = self.path + traj_name + '/' + code_name
        with open(code_file, 'rb') as f:
            data = pickle.load(f)
        
        # code = np.squeeze(data['code'], axis=0)
        # code = np.expand_dims(code[14][2], axis=0)
        # code = torch.from_numpy(code)
        # return code
        return torch.from_numpy(data['code'])
    
    def load_feature(self, i_traj, seq, rollout, skip_frame):
        traj_name = 'traj_' + str(i_traj).zfill(6)
        code_name = 'data.pkl'
        code_file = self.path + traj_name + '/' + code_name
        with open(code_file, 'rb') as f:
            data = pickle.load(f)
        data = torch.from_numpy(np.array(data['ego'][seq:seq+skip_frame*rollout:skip_frame])).type(torch.FloatTensor)
        return data

    def load_feature_code(self, i_traj, seq, rollout, skip_frame):
        traj_name = 'traj_' + str(i_traj).zfill(6)
        code_name = 'data.pkl'
        code_file = self.path + traj_name + '/' + code_name
        with open(code_file, 'rb') as f:
            data = pickle.load(f)
        data = torch.from_numpy(np.array(data['code'][seq:seq+skip_frame*rollout:skip_frame])).type(torch.LongTensor)
        return data.squeeze(dim=-1)
        

    def get_variance(self):
        # calculate variance of dataset
        # output C*H*W
        return 1.0


class CodeDataset(Dataset):
    def __init__(self, path, mode, transform=None, args=None):
        # set hyperparamter
        self.path = path
        self.mode = mode
        self.transform = transform
        
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
            name = 'traj_' + str(i_traj).zfill(6)
            dir_name = self.path + name + '/'
            n_frames = count_files(dir_name, cond='pkl')
            for seq in range(n_frames):
                self.data_list.append({'traj': i_traj, 'seq': seq})


    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        i_traj = self.data_list[idx]['traj']
        seq = self.data_list[idx]['seq']
        traj_name = 'traj_' + str(i_traj).zfill(6)
        seq_name = 'seq_' + str(seq).zfill(6)
        code_file = self.path + traj_name + '/' + seq_name + '.pkl'
        with open(code_file, 'rb') as pf:
            code = pickle.load(pf)
        
        # print(self.transform(code))
        # print(torch.unsqueeze(torch.from_numpy(code),dim=0))
        return torch.unsqueeze(torch.from_numpy(code),dim=0), 1.0

    def get_variance(self):
        # calculate variance of dataset
        # output C*H*W
        return 1.0
    

if __name__=="__main__":
    """
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
    """
    parser = argparse.ArgumentParser(description='dataloader')

    parser.add_argument('--rollout', default=8, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=10, help='interval between images in trajectory')
    parser.add_argument('--n_dec_seq', default=10, help='interval between images in trajectory')
    parser.add_argument('--jump_frame', default=10, help='interval between images in trajectory')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    PROJECT_PATH = os.path.abspath("..")
    # DATA_PATH = PROJECT_PATH + "/dataset/NGSIM/"
    DATA_PATH = PROJECT_PATH + "/dataset/NGSIM_feature/"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])
    dataset = NGSIMDataset(DATA_PATH, 'train', data_format='feature_code', prediction=True, transform=transform, args=args)
    
    data_loader = DataLoader(dataset, 
                             batch_size=1, 
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
    print(len(dataset))
    n=0
    for batch in data_loader:
        n += 1
        # print(1)
        
        # print(batch[0].shape)
        print(batch[0])
        # print(batch[1].shape)
        # # print(batch[0])
        # print(batch[1])
        # batch[0] = batch[0].cpu()
        # batch[1] = batch[1].cpu()
        # for i in range(9):
        #     print(torch.sum(batch[0][0][i] - batch[0][0][i+1] == 0))

        # print(batch[0].to(device))
        # if n > 10:
        #     break