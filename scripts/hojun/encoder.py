import torch
import torch.nn as nn
import torch.nn.functional as F
from hojun.residual import ResidualStack, ResidualStack_3d

class Encoder_bev_sep(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder_bev_sep, self).__init__()

        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=(4,5,5),
                                 stride= (2,3,3), padding=(1,2,1))
        self._conv_2 = nn.Conv3d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=(3,4,4),
                                 stride=(1,2,2), padding=1)
        self._conv_3 = nn.Conv3d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=(3,4,4),
                                 stride=(1,2,2), padding=1)
        # self._conv_4 = nn.Conv3d(in_channels=num_hiddens,
        #                          out_channels=num_hiddens,
        #                          kernel_size=3,
        #                          stride=(1,3,2), padding=(1,0,1))
        
        self._residual_stack = ResidualStack_3d(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
       
        x = self._residual_stack(x)
        

        x = torch.reshape(x, (x.shape[0], -1, x.shape[2], 1))
        x = x.contiguous()
 
        return x



class Encoder_bev(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, img_shape):
        super(Encoder_bev, self).__init__()

        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv3d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv3d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack_3d(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._weighted_sum = nn.Linear((img_shape[0]//4) * (img_shape[1]//4), 1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)

        x = self._residual_stack(x)

        x = x.flatten(start_dim=3)
        x = self._weighted_sum(x)
        return x

class Encoder_feature(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder_feature, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
