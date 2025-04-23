import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import wandb
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets

import wandb as wdb
from architecture_utils import Conv2D, Linear, _build_conv_enc, ResDenseLayer, get_act_fn

# class LinearEncoder(nn.Module):
#     def __init__(self, input_dim=784, latent_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, latent_dim)
#         )

#     def forward(self, x):
#         return self.net(x.view(x.size(0), -1))


class LinearEncoder(nn.Module):
    def __init__(self, 
                 input_dim=784, 
                 latent_dim=128, 
                 normalize=True, 
                 normalize_dim=0):
        super().__init__()
        
        self.net = nn.Sequential(
            Linear(input_dim, latent_dim, normalize=normalize, normalize_dim=normalize_dim)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
    

class ConvEncoder(nn.Module):
    def __init__(self, 
                 latent_dim=128, 
                 dataset="MNIST", 
                 use_norm=True, 
                 bias=True,
                 in_channels=None,
                 ):
        super().__init__()
        self.in_channels = 1 if dataset.endswith("MNIST") or dataset == "Omniglot" else 3
        
        if dataset in ['vH16', 'CIFAR16', 'BALLS16', 'BALLS64']:
            padding = 1
        elif dataset.endswith("MNIST") or dataset == "Omniglot":
            padding = 0
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        
        self.net = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=padding),  # [B,32,14,14] or smaller
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=padding),           # [B,64,7,7] or smaller
                nn.ReLU(),
                nn.Flatten(),
            )
        if dataset in ['vH16', 'CIFAR16', 'BALLS16', 'BALLS64']:
            dummy = torch.zeros(1, self.in_channels, 16, 16)
        elif dataset.endswith("MNIST") or dataset == "Omniglot":
            dummy = torch.zeros(1, self.in_channels, 28, 28)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        with torch.no_grad():
            flat_dim = self.net(dummy).shape[-1]

        layers = [nn.Linear(flat_dim, latent_dim)]
        
        if use_norm:
            layers.append(nn.LayerNorm(latent_dim))
        if not bias:
            layers[0].bias = None

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.net(x)
        return self.fc(x)
    

class MLPEncoder(nn.Module):
    def __init__(
            self, input_dim: int = 784,
            latent_dim: int = 128,
            expand: int = 8,
            normalize: bool = False,
            normalize_dim: int = 0,
            bias: bool = False,
        ):
        super().__init__()
        self.net = nn.Sequential(
            ResDenseLayer(input_dim, expand=expand),
            Linear(
                in_features=input_dim,
                out_features=latent_dim,
                normalize=normalize,
                normalize_dim=normalize_dim,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    
# class LinearDecoder(nn.Module):
#     def __init__(self, latent_dim=128, output_dim=784):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, output_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         return self.net(z)
    

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=784, normalize=True, normalize_dim=0):
        super().__init__()
        self.fc_dec = Linear(latent_dim, output_dim, normalize=normalize, normalize_dim=normalize_dim)
        self.net = nn.Sequential(
            # Linear(latent_dim, output_dim, normalize=normalize, normalize_dim=normalize_dim),
            self.fc_dec,
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1,size=7,normalize=True, 
                 normalize_dim=0):
        super().__init__()
        self.size=size
        # Fully connected layer to expand from latent_dim to feature map
        # self.fc_dec = nn.Linear(latent_dim, 128 * 7 * 7)
        self.fc_dec = Linear(latent_dim, 128 * self.size * self.size, normalize=normalize, normalize_dim=normalize_dim)
    
        # Transposed conv layers to upscale to 28x28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def forward(self, z):
        #z: 200 128
        x = self.fc_dec(z)  # [B, 128*7*7]
        #x: 200 6272
        x = x.view(-1, 128, self.size, self.size)  # reshape to [B, 128, 7, 7]
        #200 1 28 28
        x = self.deconv(x)  # [B, 1, 28, 28]3 16 16
        return x
    

class MLPDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            output_dim: int = 784,
            normalize: bool = False,
            normalize_dim: int = 0,
            bias: bool = False,
            activation_fn: str = "swish",
        ):
        super().__init__()
        self.net = nn.Sequential(
            Linear(
                in_features=latent_dim,
                out_features=output_dim,
                normalize=normalize,
                normalize_dim=normalize_dim,
                bias=bias,
            ),
            get_act_fn(activation_fn),
            ResDenseLayer(output_dim),
            get_act_fn(activation_fn),
            nn.Linear(in_features=output_dim, out_features=output_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)

# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim=128, n_ch=32, normalize=False, bias=True):
#         super().__init__()
#         spat_dim = 7
#         self.shape = (-1, n_ch * 8, spat_dim, spat_dim)

#         self.fc_dec = nn.Linear(latent_dim, n_ch * 8 * spat_dim * spat_dim, bias=bias)

#         self.dec = nn.Sequential(
#             nn.ReLU(),
#             nn.ConvTranspose2d(n_ch * 8, n_ch * 4, kernel_size=4, stride=2, padding=1),  # [B, n_ch*4, 4, 4]
#             nn.ReLU(),
#             nn.ConvTranspose2d(n_ch * 4, n_ch * 2, kernel_size=4, stride=2, padding=1),  # [B, n_ch*2, 8, 8]
#             nn.ReLU(),
#             nn.ConvTranspose2d(n_ch * 2, 1, kernel_size=4, stride=2, padding=1),         # [B, 1, 16, 16] → 可调整为 28x28
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         x = self.fc_dec(z)
#         x = x.view(*self.shape)
#         return self.dec(x)


    

