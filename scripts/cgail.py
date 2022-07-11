from copy import deepcopy
import torch
import numpy as np
import yaml
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse

import torch.nn
from model import Discriminator, SACCore
from simple_env import SimpleEnv
from utils import *
from replay_buffer import ReplayBuffer

PROJECT_PATH = os.path.abspath("..")
POLICY_PATH = PROJECT_PATH + "/policy/"
DATASET_PATH = PROJECT_PATH + "/dataset/"
YAML_PATH = PROJECT_PATH + "/yaml/"

if __name__ == "__main__":
    init_time_ = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description='GAIL')
    parser.add_argument('--exp_name', default='dynamics_6', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM_feature', help='dataset name')
    parser.add_argument('--data_type', default='train', help='name of the dataset what we label')
    parser.add_argument('--policy_file', default='best', help='policy file name')
    parser.add_argument('--rollout', default=10, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--feature_dim', default=5, help='interval between images in trajectory')
    parser.add_argument('--learning_rate', default=5e-6, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=256, help='interval between images in trajectory')
    
    args = parser.parse_args()
    
    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    EXP_PATH = POLICY_PATH + args.exp_name + "/"
    DATA_PATH = DATASET_PATH + args.data_name + "/" + args.data_type + "/"
    POLICY_FILE = EXP_PATH + args.policy_file + ".pt"
    SAVE_PATH = DATA_PATH + args.data_type + "/"
    MODEL_PATH = EXP_PATH + "best.pt"

    make_dir(EXP_PATH)

    if args.env_name == 'SimpleEnv':
        env_ = SimpleEnv(args)
    else:
        print("NotImplementError")
        exit()
    
    obs_dim_, act_dim_ = env_.get_dim()
    
    # load NGSIM dataset

    

    # set random seed
    np.random.seed(args.seed)

    # create model and replay buffer
    ac_ = SACCore(obs_dim_, act_dim_, [64, 32], args.learning_rate, args.act_limit).to(device)
    ac_tar_ = deepcopy(ac_)
    D_ = []
    for i in range(args.n_class):
        D_.append(Discriminator(obs_dim_, act_dim_, [64, 32], args.learning_rate).to(device))
    
    # load NGSIM dataset
    
    
    replay_buffer_ = ReplayBuffer(obs_dim_, act_dim_, args.replay_size, device)

    # target network doesn't have gradient
    for p in ac_tar_.parameters():
        p.requires_grad = False
    
    def compute_loss_d(e_data, p_data):
        eo = tensor_from_numpy(e_data['obs'], device_)
        ea = tensor_from_numpy(e_data['act'], device_)
        po, pa = p_data['obs'], p_data['act']

        e_prob = D_(eo, ea)
        p_prob = D_(po, pa)

        loss = - torch.mean(torch.log(1.0 - e_prob)) - torch.mean(torch.log(p_prob))

        learner_acc = ((p_prob >= 0.5).float().mean().item())
        expert_acc = ((e_prob < 0.5).float().mean().item())

        print("Expert: %.5f%% | Learner: %.f%%" % (expert_acc * 100, learner_acc * 100))
        flag = True
        if expert_acc > args.acc_threshold and learner_acc > args.acc_threshold:
            flag = False

        d_info = {'d_loss': loss.cpu().item(), 'flag': flag}

        return loss, d_info

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        with torch.no_grad():
            r = D_.get_reward(o, a)
        q1 = ac_.q1(o,a)
        q2 = ac_.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac_.pi(o2)

            # Target Q-values
            q1_pi_tar = ac_tar_.q1(o2, a2)
            q2_pi_tar = ac_tar_.q2(o2, a2)
            q_pi_tar = torch.min(q1_pi_tar, q2_pi_tar)
            backup = r + args.gamma * (1 - d) * (q_pi_tar - args.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # define pi loss function
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac_.pi(o)
        q1_pi = ac_.q1(o, pi)
        q2_pi = ac_.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (args.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update_D(e_data, p_data):
        D_.optimizer.zero_grad()
        loss_d, d_info = compute_loss_d(e_data, p_data)
        loss_d.backward()
        D_.optimizer.step()

        return d_info

    # define update function
    def update(data):
        ac_.q1.optimizer.zero_grad()
        ac_.q2.optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        ac_.q1.optimizer.step()
        ac_.q2.optimizer.step()
        
        for p in ac_.q1.parameters():
            p.requires_grad = False
        for p in ac_.q2.parameters():
            p.requires_grad = False

        ac_.pi.optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        ac_.pi.optimizer.step()

        for p in ac_.q1.parameters():
            p.requires_grad = True
        for p in ac_.q2.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(ac_.parameters(), ac_tar_.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(args.polyak)
                p_targ.data.add_((1 - args.polyak) * p.data)

    def get_action(o, remove_grad=True):
        a = ac_.act(torch.unsqueeze(torch.as_tensor(o, dtype=torch.float32), dim=0).to(device=device))
        if remove_grad:
            return a.detach().cpu().numpy()
        return a

    # for evaluation in each epoch
    def test_agent():
        tot_ep_ret = 0.0
        for _ in range(args.n_log_epi):
            o, d, ep_ret, ep_len = env_.reset(), False, 0, 0
            while not(d or (ep_len == args.epi_len_)):
                # Take deterministic actions at test time 
                o, r, d, _ = env_.step(get_action(o))
                ep_ret += r
                ep_len += 1
            tot_ep_ret += ep_ret
        return tot_ep_ret / args.n_log_epi

    total_steps = args.steps_per_epoch * args.epochs
    start_time = time.time()
    o, ep_ret, ep_len = env_.reset(), 0, 0

    ts_axis = []
    rt_axis = []
    max_avg_rt = -1000.0
    D_flag = True

    # Main loop: collect experience in env and update/log each epoch
    
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > args.start_steps:
            a = get_action(o)
        else:
            a = env_.action_space.sample()

        # Step the env
        o2, r, d, _ = env_.step(a)
        # r = D_.get_reward(o,a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == args.epi_len else d

        # Store experience to replay buffer
        replay_buffer_.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == args.epi_len):
            o, ep_ret, ep_len = env_.reset(), 0, 0

        # Update handling
        if D_flag:
            if t >= args.update_after and t % args.update_every == 0:
                for j in range(args.update_every):
                    e_batch = sample_demo(exp_demo_, batch_size_)
                    p_batch = replay_buffer_.sample_batch(batch_size_)
                    info = update_D(e_batch, p_batch)
                    D_flag = info['flag']
        if t >= update_after_ and t % update_every_ == 0:
            for j in range(update_every_):
                batch = replay_buffer_.sample_batch(batch_size_)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch_ == 0:
            epoch = (t+1) // steps_per_epoch_
            avg_rt = test_agent()
            ts_axis.append(t+1)
            rt_axis.append(avg_rt)
            if epoch % save_interval_ == 0:
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_"+str(epoch).zfill(3)+".pt")
            if max_avg_rt < avg_rt:
                max_avg_rt = avg_rt
                torch.save(ac_, POLICY_PATH + exp_name_ + "/ac_best.pt")    
            print("[%.3f] Epoch: %d, Timesteps: %d, AvgEpReward: %.3f" %(time.time() - init_time_, epoch, t+1, avg_rt))
            if plot_rendering_:
                plt.plot(ts_axis, rt_axis)
                plt.pause(0.001)
