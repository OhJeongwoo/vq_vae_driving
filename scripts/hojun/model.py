import torch
import torch.nn as nn
import torch.nn.functional as F
from hojun.encoder import Encoder_bev, Encoder_feature, Encoder_bev_sep
from hojun.decoder import Decoder_bev, Decoder_feature, Decoder_bev_sep
from hojun.quantizer import VectorQuantizerEMA, VectorQuantizer

class Model(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, img_shape, output_padding, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self.num_hiddens = num_hiddens

        self.bev_encoder = Encoder_bev(in_channels[0], num_hiddens[0],
                                num_residual_layers[0], 
                                num_residual_hiddens[0], img_shape)

        self.feature_encoder = Encoder_feature(in_channels[1], num_hiddens[1],
                                num_residual_layers[1], 
                                num_residual_hiddens[1])

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self.bev_decoder = Decoder_bev(embedding_dim[0], num_hiddens[0],
                                num_residual_layers[0], 
                                num_residual_hiddens[0], img_shape, output_padding[0])
        self.feature_decoder = Decoder_feature(embedding_dim[1], num_hiddens[1],
                                num_residual_layers[1], 
                                num_residual_hiddens[1], output_padding[1])

    def forward(self, bev, feature):
        z_bev = self.bev_encoder(bev)
        z_feature = self.feature_encoder(feature)

        z = torch.cat((z_bev, z_feature), dim=1)
        #TODO : check vectorquantizer / concat z_bev and z_feature / check loss / start train
        loss, quantized, perplexity, _ = self._vq_vae(z)
        
        code_bev, code_feature = torch.split(quantized,[self.num_hiddens[0], self.num_hiddens[1]], dim=1)
        
        bev_recon = self.bev_decoder(code_bev)
        feature_recon = self.feature_decoder(code_feature)

        return loss, bev_recon, feature_recon, perplexity


class Model_sep(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, output_padding, commitment_cost, img_shape, dim_ratio, channel_type, decay=0):
        super(Model_sep, self).__init__()
        
        self.num_hiddens = num_hiddens

        self.bev_encoder = Encoder_bev_sep(len(channel_type), num_hiddens[0],
                                num_residual_layers[0], 
                                num_residual_hiddens[0])

        # self.feature_encoder = Encoder_feature(in_channels[1], num_hiddens[1],
        #                         num_residual_layers[1], 
        #                         num_residual_hiddens[1])
        latent_shape = (img_shape[0] // dim_ratio, img_shape[1] // dim_ratio)
        
        channel_increase_ratio = (img_shape[0] // dim_ratio) * (img_shape[1] // dim_ratio)
        self._pre_vq_conv_bev = nn.Conv2d(in_channels=num_hiddens[0] * channel_increase_ratio, 
                                      out_channels=embedding_dim[0],
                                      kernel_size=1, 
                                      stride=1)
        # self._pre_vq_conv_feature = nn.Conv2d(in_channels=num_hiddens[1], 
        #                               out_channels=embedding_dim[1],
        #                               kernel_size=1, 
        #                               stride=1)
    
        if decay > 0.0:
            self._vq_vae_bev = VectorQuantizerEMA(num_embeddings[0], embedding_dim[0], 
                                              commitment_cost, decay)
            # self._vq_vae_feature = VectorQuantizerEMA(num_embeddings[1], embedding_dim[1],
            #                                     commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self.bev_decoder = Decoder_bev_sep(embedding_dim[0] // channel_increase_ratio, num_hiddens[0],
                                num_residual_layers[0], 
                                num_residual_hiddens[0], output_padding[0], latent_shape, channel_type)
        # self.feature_decoder = Decoder_feature(embedding_dim[1], num_hiddens[1],
        #                         num_residual_layers[1], 
        #                         num_residual_hiddens[1], output_padding[1])

    def forward(self, bev):
        z_bev = self.bev_encoder(bev)
        # z_feature = self.feature_encoder(feature)
        # print("encoder output: ", z_bev.shape)
        z_bev = self._pre_vq_conv_bev(z_bev)
        # z_feature = self._pre_vq_conv_feature(z_feature)
        # print("before quantization:", z_bev.shape)

        loss_bev, quantized_bev, perplexity_bev, _ = self._vq_vae_bev(z_bev)
        # loss_feature, quantized_feature, perplexity_feature, _ = self._vq_vae_feature(z_feature)
        # print("after quantization: ", quantized_bev.shape)
        bev_recon = self.bev_decoder(quantized_bev)
        # feature_recon = self.feature_decoder(quantized_feature)

        # loss = loss_bev + loss_feature
        loss = loss_bev
        # perplexity = perplexity_bev + perplexity_feature
        perplexity = perplexity_bev
        return loss, bev_recon, perplexity

class Model_feature(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, output_padding, commitment_cost, decay=0):
        super(Model_feature, self).__init__()
        
        self.num_hiddens = num_hiddens


        self.feature_encoder = Encoder_feature(in_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self.feature_decoder = Decoder_feature(embedding_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, output_padding)

    def forward(self, feature):
        z= self.feature_encoder(feature)

        z = self._pre_vq_conv(z)

        loss, quantized, perplexity, _ = self._vq_vae(z)
                
        feature_recon = self.feature_decoder(quantized)

        return loss, feature_recon, perplexity