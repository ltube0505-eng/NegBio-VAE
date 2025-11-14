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
from architecture_utils import (Conv2D, Linear, ResDenseLayer, _build_conv_enc,
                                get_act_fn)

# To ensure fair comparisons, 
# we use the same MLP, linear, and convolutional architectures 
# as those in the Poisson VAE repository.

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
        self.in_channels = 1 if dataset.endswith("MNIST") or dataset == "fmnist" else 3
        
        if dataset in ['vH16', 'CIFAR16', 'BALLS16', 'BALLS64']:
            padding = 1
        elif dataset.endswith("MNIST") or dataset in ["fmnist", 'SVHN']:
            padding = 0
        elif dataset in ["CelebA", "CelebA64", "FFHQ"]:
            padding = 1   
        elif dataset in ["CelebAHQ"]:
            padding = 1
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.net = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=padding), 
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=padding),          
                nn.ReLU(),
                nn.Flatten(),
            )
        if dataset in ['vH16', 'CIFAR16', 'CIFAR10','BALLS16', 'BALLS64']:
            dummy = torch.zeros(1, self.in_channels, 16, 16)
        elif dataset.endswith("MNIST") or dataset == "fmnist":
            dummy = torch.zeros(1, self.in_channels, 28, 28)
        elif dataset == 'SVHN':
            dummy = torch.zeros(1, self.in_channels, 32, 32)
        elif dataset in ["CelebA", "CelebA64", "FFHQ"]:
            dummy = torch.zeros(1, self.in_channels, 64, 64)
        elif dataset == "CelebAHQ":
            dummy = torch.zeros(1, self.in_channels, 128, 128)
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


import torchvision.models as models
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim=128, fc_hidden1=1024,fc_hidden2=768) -> None:
        super().__init__()

        #cnn architecture
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)

        self.fc_hidden1,self.fc_hidden2, self.latent_dim = fc_hidden1, fc_hidden2, latent_dim
        resnet = models.resnet152(pretrained=True)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1= nn.Linear(resnet.fc.in_features,self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1,momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2,momentum=0.01)
        self.fc_z = nn.Linear(self.fc_hidden2,self.latent_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.bn2(x))
        z = self.fc_z(x)
        return z



class ResNetDecoder(nn.Module):
    def __init__(self,latent_dim=128,fc_hidden1=1024,fc_hidden2=1024) -> None:
        super().__init__()
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.latent_dim = latent_dim
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)     
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)

        self.fc1 = nn.Linear(self.latent_dim,self.fc_hidden2)
        self.fc_bn1 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc2 = nn.Linear(self.fc_hidden2, 64*4*4)
        self.fc_bn2 = nn.BatchNorm1d(64*4*4)
        self.relu = nn.ReLU(inplace=True)


        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32,momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(            
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    
        )

    def forward(self,x):
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x))).view(-1,64,4,4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.nn.functional.interpolate(x, size=(224,224),mode='bilinear')
        return x


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

    

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=784, normalize=True, normalize_dim=0, tanh=False):
        super().__init__()
        self.fc_dec = Linear(latent_dim, output_dim, normalize=normalize, normalize_dim=normalize_dim)
        if tanh:
            self.net = nn.Sequential(
                self.fc_dec,
                nn.Tanh(),
            )
        else:

            self.net = nn.Sequential(
                self.fc_dec,
                nn.Sigmoid(),
            )

    def forward(self, z):
        return self.net(z)


class MLPDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            output_dim: int = 784,
            normalize: bool = False,
            normalize_dim: int = 0,
            bias: bool = False,
            activation_fn: str = "swish",
            tanh=False
        ):
        super().__init__()
        self.fc_dec = Linear(
                in_features=latent_dim,
                out_features=output_dim,
                normalize=normalize,
                normalize_dim=normalize_dim,
                bias=bias,
            )
        if tanh:
            self.net = nn.Sequential(
                self.fc_dec,
                get_act_fn(activation_fn),
                ResDenseLayer(output_dim),
                get_act_fn(activation_fn),
                nn.Linear(in_features=output_dim, out_features=output_dim, bias=True),
                nn.Tanh(),
            )
        else:

            self.net = nn.Sequential(
                self.fc_dec,
                get_act_fn(activation_fn),
                ResDenseLayer(output_dim),
                get_act_fn(activation_fn),
                nn.Linear(in_features=output_dim, out_features=output_dim, bias=True),
                nn.Sigmoid(),
            )
    def forward(self, z):
        return self.net(z)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=1,size=7,normalize=True, 
                 normalize_dim=0, tanh=False):
        super().__init__()
        self.size=size
        self.fc_dec = Linear(latent_dim, 128 * self.size * self.size, normalize=normalize, normalize_dim=normalize_dim)
    
        if tanh:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),   
                nn.Tanh(),
            )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
                nn.ReLU(),
                nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  
                nn.Sigmoid(),
            )


    def forward(self, z):
        x = self.fc_dec(z)  
        x = x.view(-1, 128, self.size, self.size) 
        x = self.deconv(x)
        return x


    

