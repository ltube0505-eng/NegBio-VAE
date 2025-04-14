import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance


class LinearEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
    

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # [B,1,28,28] → [B,32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B,64,7,7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
    
    
class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
    

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B,32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # [B,1,28,28]
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
    



    

