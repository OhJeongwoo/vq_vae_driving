import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
import argparse


class SimpleEnv:
    def __init__(self, args):
        # load NGSIM feature
        PROJECT_PATH = os.path.abspath("..")
        DATASET_PATH = PROJECT_PATH + "/dataset/"
        DATA_PATH = DATASET_PATH + "GAIL_data.pkl"

        with open(DATA_PATH, 'rb') as f:
            self.data = pickle.load(f)

        self.num_data = len(self.data)

        self.jerk_weight = args.jerk_weight
        self.dt = 0.1
        self.rollout = args.rollout # 16
        # save dataset info by rollout and skip frame scale
        
        # initialize state
        self.reset()
        # traj_info = self.data[0]
        # self.traj = traj_info["state"]
        # self.final_state = traj_info["action"]
        # self.cluster = traj_info["cluster"]
        # self.car_id = traj_info["car_id"]
        # self.traj_len = self.traj.shape[0]
        # self.index = 0
        # # return initial state
        # self.state = self.traj[0]
        # self.previus_a = self.state[3]
        # self.previous_w = self.state[4]

        # self.reset()
        return self.get_state()
    

    def reset(self):
        # select random trajectory
        index = random.randrange(self.num_data)

        traj_info = self.data[index]
        self.traj = traj_info["state"]
        self.cluster = traj_info["cluster"]
        self.car_id = traj_info["car_id"]
        self.traj_len = self.traj.shape[0]
        self.steps = 0
        # return initial state
        self.state = self.traj[0]

    
    def step(self, action):
        # calculate next state
        a = action[0]
        w = action[1]

        pv = self.state[2]
        px = self.state[0]
        py = self.state[1]
        pa = self.state[-1][3]
        pw = self.state[-1][4]
        ph = self.state[5]

        v = pv + a * self.dt
        h = ph + w * self.dt
        x = px + v * math.cos(h) * self.dt
        y = py + v * math.sin(h) * self.dt

        self.index += 1
        # self.state = self.traj[self.index]
        self.state = torch.tensor([x,y,v,a,w,h])

        # calculate reward
        reward = self.jerk_weight * F.mse_loss(a, self.previus_a) + F.mse_loss(w, self.previous_w)
        
        self.previus_a = a
        self.previous_w = w

        done = False
        if self.steps == self.max_steps:
            done = True
        return reward


    def get_dim(self):
        # calculate state dim and action dim

        return 1,1

    
    def get_state(self):
        return self.state[:,:5]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--jerk_weight', default=0.1, type=float, help='jerk hyperparameter')
    args = parser.parse_args()
    env = SimpleEnv(args)
    print(env.state)
    env.reset()
    print(env.state)
    env.reset()
    
    print(env.state)

    score = env.step(torch.tensor([0.5,0.0001]))

    print(env.state)