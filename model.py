import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import wandb
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets
import torch.nn.functional as F
import wandb as wdb
from distribution import NegBinomial, Poisson, Categorical
from utils import find_critical_ids, find_last_contiguous_zeros, tonp
from torch.distributions import RelaxedOneHotCategorical

class GenericVAE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 latent_dim: int=128,
                 dist_type: str = 'negbio',
                 dist_class = None,
                 reparam_type = "gamma",
                 max_count = 15,
                 tau = 1.0,
                 **kwargs
                 ):
        super(GenericVAE, self).__init__()
        assert dist_type in ['poisson', 'negbio', 'category']
        self.dist_type = dist_type
        # self.dist_class = dist_class
        self.latent_dim = latent_dim 
        self.reparam_type = reparam_type
        self.max_count = max_count
        self.tau = tau
        
        self.encode = encoder 
        self.decode = decoder 
        self.t = 1.0

        if dist_type == 'poisson':
            self.prior = nn.Parameter(torch.zeros((1, latent_dim)))
        elif dist_type == 'negbio':
            self.dist_class = NegBinomial(
                            reparam_type=reparam_type,
                            max_count=max_count,
                            tau=tau,
                        )
            self.log_r_prior = nn.Parameter(torch.zeros((1, latent_dim)))
            self.logit_p_prior = nn.Parameter(torch.zeros((1, latent_dim)))
        elif dist_type == 'category':
            self.dist_class = None
            
        else:
            raise NotImplementedError
        
    def mse_loss(self, x, y):
        return ((y - x)**2).sum(-1).mean()
        
    def forward(self, x):
        validation = not torch.is_grad_enabled()

        if self.dist_type == "poisson":
            du = self.encode(x).clamp(None, 5)
            dist = Poisson((du + self.prior.clamp(None, 5)).clamp(None, 5), self.t)
            z = dist.rsample(hard=validation)
            y = self.decode(z)
            return dist, du, z, y

        elif self.dist_type == "negbio":
            logit_p = self.encode(x).clamp(-5, 5)
            z = self.dist_class.rsample(self.log_r_prior, logit_p, self.t, hard=validation)             
            y = self.decode(z)
            return self.dist_class, logit_p, z, y
        elif self.dist_type == "category":
            logit_p = self.encode(x) 
            self.dist_class = Categorical(logits=logit_p)
            z = self.dist_class.rsample()
            #z_one_hot = F.one_hot(z, num_classes=self.num_classes).float()
            y = self.decode(z)
            return self.dist_class, logit_p, z, y
        else:
            raise NotImplementedError
        

    def find_dead_neurons(self, frac: int = 8):
        norms = tonp(torch.linalg.vector_norm(
            self.decode.fc_dec.weight, dim=0))  # assumes decoder has `fc_dec`
        
        eps = np.finfo(norms.dtype).eps
        norms = np.maximum(norms, eps)
        log_norms = np.log(norms)

        # finds large contiguous gap
        bins = np.linspace(
            start=np.nanmin(log_norms),
            stop=np.nanmax(log_norms),
            num=len(log_norms) * 10
        )
        hist, _ = np.histogram(log_norms, bins=bins)
        median_idx = np.digitize(
            x=np.median(log_norms),
            bins=bins,
        ) - 1

        idx = find_last_contiguous_zeros( 
            mask=hist[:median_idx] > 0,
            w=len(log_norms) * 2,
        )
        dead1 = log_norms < bins[idx]

        # finds smallest and largest outliers
        bins = np.linspace(
            start=np.nanmin(log_norms),
            stop=np.nanmax(log_norms),
            num=len(log_norms) // frac
        )
        hist, _ = np.histogram(log_norms, bins=bins)
        i, j = find_critical_ids(hist > 0)  

        dead2 = np.logical_or(
            log_norms < bins[i],
            log_norms > bins[j],
        )
        return dead1 | dead2
    
     
     
    







