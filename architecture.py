import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from architecture_utils import Linear, Conv2D, _build_conv_enc


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
                 in_channels=1):
        super().__init__()

        if dataset in ['vH16', 'CIFAR16', 'BALLS16', 'BALLS64']:
            padding = 1
        elif dataset.endswith("MNIST") or dataset == "Omniglot":
            padding = 0  
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=padding),  # [B,32,14,14] or smaller
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=padding),           # [B,64,7,7] or smaller
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy = torch.zeros(1, in_channels, 28, 28)
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
        self.net = nn.Sequential(
            Linear(latent_dim, output_dim, normalize=normalize, normalize_dim=normalize_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1):
        super().__init__()

        # Fully connected layer to expand from latent_dim to feature map
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        # Transposed conv layers to upscale to 28x28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)  # [B, 128*7*7]
        x = x.view(-1, 128, 7, 7)  # reshape to [B, 128, 7, 7]
        x = self.deconv(x)  # [B, 1, 28, 28]
        return x
    


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


    

