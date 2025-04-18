import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
import matplotlib.pyplot as plt 
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from distribution import NegBinomial, Poisson, NegBinomial_Gamma



# class NBVAE(nn.Module):
#     def __init__(self):
#         super(NBVAE, self).__init__()
#         self.encode = nn.Sequential(
#             nn.Linear(784, 128),
#         )
#         self.decode = nn.Sequential(
#             nn.Linear(128, 784),
#             nn.Sigmoid(),
#         )
#         self.log_r_prior = nn.Parameter(torch.zeros((1, 128)))
#         self.logit_p_prior = nn.Parameter(torch.zeros((1, 128)))
#         self.t = 1.0 #temperature
#     def forward(self, x):
#         validation = not torch.is_grad_enabled()
#         logit_p = self.encode(x).clamp(-5, 5)
#         dist = NegBinomial(self.log_r_prior, logit_p, self.t)
#         z = dist.rsample(hard=validation)
#         y = self.decode(z)
#         return dist, logit_p, z, y
    


# class PVAE(nn.Module):
#     def __init__(self):
#         super(PVAE, self).__init__()
#         self.encode = nn.Sequential(
#             nn.Linear(784, 128),
#         )
#         self.decode = nn.Sequential(
#             nn.Linear(128, 784),
#             nn.Sigmoid(),
#         )
#         self.prior = nn.Parameter(torch.zeros((1, 128)))
#         self.t = 1.0 #temperature
#     def forward(self, x):
#         validation = not torch.is_grad_enabled()
#         du = self.encode(x).clamp(None, 5)
#         dist = Poisson((du + self.prior.clamp(None, 5)).clamp(None, 5), self.t)
#         z = dist.rsample(hard=validation)
#         y = self.decode(z)
#         return dist, du, z, y
    


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
        assert dist_type in ['poisson', 'negbio']
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

        else:
            raise NotImplementedError
        
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
            if self.reparam_type == "gamma":
                # dist = NegBinomial(self.reparam_type, self.max_count, self.tau)
                # z = self.dist_class.rsample(self.log_r_prior, logit_p, self.t, hard=validation)
                dist = NegBinomial_Gamma(self.log_r_prior, logit_p, self.t)
                z = dist.rsample(hard=validation)
            elif self.reparam_type == "gumbel":
                #dist = NegBinomial(self.reparam_type, self.max_count, self.tau)
                z = self.dist_class.rsample(self.log_r_prior, logit_p, self.t, hard=validation)
                # print(z[0, :10])

            # print(f"[Gumbel] z mean: {z.mean().item():.2f}, std: {z.std().item():.2f}")
             
            y = self.decode(z)
            return self.dist_class, logit_p, z, y
        
        else:
            raise NotImplementedError
    
     
     
    







