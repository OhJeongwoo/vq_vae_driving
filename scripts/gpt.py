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
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import json

from dataloader import NGSIMDataset

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu() # upper triangular part of a matrix(2-D)
    return subsequent_mask

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, self_attn_mask):
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(self_att_outputs)
        ffn_outputs = self.layer_norm3(self_att_outputs + ffn_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob

class Decoder(nn.Module):
    def __init__(self, config, start):
        super().__init__()
        self.config = config
        self.start = start
        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq, self.config.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs):
        if self.start:
            positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=torch.long).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous()
        
            # (bs, n_dec_seq, d_hidn)
            # dec_outputs = self.dec_emb(dec_inputs)
            
            dec_outputs = dec_inputs 
            dec_outputs += self.pos_emb(positions)
        
            # (bs, n_dec_seq, n_dec_seq)
            dec_attn_decoder_mask = get_attn_decoder_mask(torch.ones((dec_inputs.shape[0], dec_inputs.shape[1])))
            # (bs, n_dec_seq, n_dec_seq)
            dec_self_attn_mask = torch.gt(dec_attn_decoder_mask, 0)
        else:
            dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs[:,:,0])
            dec_self_attn_mask = torch.gt(dec_attn_decoder_mask, 0)
            dec_outputs = dec_inputs

        self_attn_probs = []
        for layer in self.layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, self_attn_prob = layer(dec_outputs, dec_self_attn_mask)
            self_attn_probs.append(self_attn_prob)
        # (bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)]
        return dec_outputs, self_attn_probs

class GPT(nn.Module):
    def __init__(self, config, start):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config, start)
    
    def forward(self, dec_inputs):
        # (bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)]
        dec_outputs, dec_self_attn_probs = self.decoder(dec_inputs)
        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_dec_seq, n_dec_seq)]
        return dec_outputs, dec_self_attn_probs
    
    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class GPT_pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.s_network = GPT(self.config, start=True)
        self.a_network = GPT(self.config, start = False)

        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_dec_vocab, bias=False)
        self.projection_lm.weight = self.s_network.decoder.dec_emb.weight
        
    def forward(self, dec_inputs):
        
        output_s, _ = self.s_network(dec_inputs)

        output_a, _ = self.a_network(output_s)

        logits_lm = self.projection_lm(output_a)

        action_score = nn.Softmax(dim=-1)(logits_lm)


        return action_score



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='pretrain GPT')
    parser.add_argument('--mode', default='train', help='execution mode')
    parser.add_argument('--exp_name', default='NGSIM_220524', help='experiment name')
    parser.add_argument('--data_name', default='NGSIM', help='dataset name')
    parser.add_argument('--rollout', default=20, help='rollout length of trajectory')
    parser.add_argument('--skip_frame', default=1, help='interval between images in trajectory')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--n_updates', default=5000, type=int, help='the number of updates in training')
    args = parser.parse_args()

    print("==============================Setting==============================")
    print(args)
    print("===================================================================")
    
    PROJECT_PATH = os.path.abspath("..")
    POLICY_PATH = PROJECT_PATH + "/policy/"
    DATASET_PATH = PROJECT_PATH + "/dataset/"
    EXP_PATH = "gpt_log/"
    DATA_PATH = DATASET_PATH + args.data_name + "/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])

    # dataset = CarlaDataset(DATA_PATH, args.data_type, transform, args)
    
    train_data = NGSIMDataset(DATA_PATH, 'train', data_format='code', prediction=True, transform=transform, args=args)
    valid_data = NGSIMDataset(DATA_PATH, 'valid', data_format='code', prediction=True, transform=transform, args=args)
    test_data = NGSIMDataset(DATA_PATH, 'test', data_format='code', prediction=True, transform=transform, args=args)

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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = Config({
    "n_dec_vocab": 512,
    "n_dec_seq": 20,
    "n_layer": 6,
    "d_hidn": 177,
    "i_pad": 0,
    "d_ff": 1024,
    "n_head": 2,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12,
    "learning_rate" : 4e-5,
    "batch_size" : 64,
    "n_epochs" : 1000
    })

    codebook_size = config["n_dec_vocab"]
    batch_size = config["batch_size"]
    sequence_length = config["n_dec_seq"]

    model = GPT_pretrain(config)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.99), amsgrad=False)
    model.train()
    loss_log = []

    if args.mode ==  "train":
        optimal_recon = None
        print("lets start")
        for i in xrange(config.n_epochs):

            # inputs = torch.randint(0, codebook_size, (batch_size, sequence_length))
            # next_inputs = torch.randint(0, codebook_size, (batch_size, sequence_length))
            # inputs = inputs.to(device)
            # next_inputs = next_inputs.to(device)
            (data, label) = next(iter(train_loader))
            data = data.to(device)
            label = label['label'].to(device)
            print(data.shape)
            print(label.shape)


            optimizer.zero_grad()

            action_score = model.forward(data)
            # action_score = model.forward(inputs)

 
            loss = F.cross_entropy(action_score.view(-1, codebook_size), next_inputs.view(-1))

            loss.backward()

            optimizer.step()
        
            loss_log.append(loss.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(loss_log[-100:]))
                model.s_network.save(i+1, loss, EXP_PATH + "f_s/model_" + str(i+1)+".pt")
                model.a_network.save(i+1, loss, EXP_PATH + "f_a/model_" + str(i+1)+".pt")
                print("wwww")
                if optimal_recon is None or optimal_recon > np.mean(loss_log[-100:]):
                    model.s_network.save(i+1, loss, EXP_PATH + "f_s/best.pt")
                    model.a_network.save(i+1, loss, EXP_PATH + "f_a/best.pt")
                    optimal_recon = np.mean(loss_log[-100:])
                

                f = plt.figure(figsize=(16,8))
                ax = f.add_subplot(1,1,1)
                ax.plot(loss_log, label='training')
                # ax.plot(val_recon_error_smooth, label='validation')
                ax.legend
                ax.set_yscale('log')
                ax.set_title('Smoothed NMSE.')
                ax.set_xlabel('iteration')

                plt.savefig(EXP_PATH+"training_logs/" +str(i+1).zfill(3) + ".png")





