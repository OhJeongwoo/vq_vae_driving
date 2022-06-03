import torch
import torch.nn as nn
import torch.nn.functional as F
from hojun.residual import Residual, ResidualStack, ResidualStack_3d

class Decoder_bev_sep(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_padding, latent_shape, channel_type):
        super(Decoder_bev_sep, self).__init__()
        self.latent_shape = latent_shape
        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack_3d(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose3d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens,
                                                kernel_size=(3,4,4), 
                                                stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,0))
        
        self._conv_trans_2 = nn.ConvTranspose3d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=(3,4,4), 
                                                stride=(1,2,2), padding=1, output_padding=(0,1,0))
        self._conv_trans_3 = nn.ConvTranspose3d(in_channels=num_hiddens//2, 
                                                out_channels=len(channel_type),
                                                kernel_size=(4,5,5), 
                                                stride=(2,3,3), padding=(1,1,1), output_padding=(0,0,0))
        # self._conv_trans_4 = nn.ConvTranspose3d(in_channels=num_hiddens//4, 
        #                                         out_channels=3,
        #                                         kernel_size=5, 
        #                                         stride=(1,3,2), padding=2, output_padding=(0,2,1))

    def forward(self, inputs):
        x = torch.reshape(inputs, (inputs.shape[0], -1, inputs.shape[2], self.latent_shape[0], self.latent_shape[1]))
        x = x.contiguous()

        # print("decoder_input:", x.shape)
        x = self._conv_1(x)
        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)
        # print("first dec: ", x.shape)

        x = self._conv_trans_2(x)
        # print("sec dec: ", x.shape)

        x = self._conv_trans_3(x)
        # print("third dec: ", x.shape)
        return x


class Decoder_bev(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, img_shape, output_padding):
        super(Decoder_bev, self).__init__()
        self._img_shape = img_shape
        self._inverse_weighted_sum = nn.Linear(1, (img_shape[0]//4) * (img_shape[1]//4))

        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack_3d(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose3d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1, output_padding=output_padding[:3])
        
        self._conv_trans_2 = nn.ConvTranspose3d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1, output_padding=output_padding[3:])

    def forward(self, inputs):

        x = self._inverse_weighted_sum(inputs)
        x = x.view(x.shape[:3] + (self._img_shape[0]//4, self._img_shape[1]//4))

        x = self._conv_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class Decoder_feature(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_padding):
        super(Decoder_feature, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1, output_padding = output_padding[:2])
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=4,
                                                kernel_size=4, 
                                                stride=2, padding=1, output_padding= output_padding[2:])

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)