import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
import torch
from scipy.stats import poisson
import numpy as np

class Poisson:
    # This code is copied from Poisson-VAE
    def __init__(self, log_rate, t=0.0):
        self.log_rate = log_rate
        self.rate = torch.exp(
            self.log_rate.clamp(None, 5)
        ) + 1e-6
        self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
        self.t = t
    def rsample(self, hard: bool = False):
        x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  # inter-event times
        times = torch.cumsum(x, dim=0)  # arrival times of events
        indicator = times < 1.0 # did events arrive before the end of the time interval
        if not (hard or self.t == 0): # soften the indicator function
            indicator = torch.sigmoid((1.0 - times) / self.t)
        return indicator.sum(0).float()
    def kl(self, prior, du):
        #"prior" argument referes to log rate of prior
        #equation is r * (1 - dr + dr * log(dr))
        r = torch.exp(prior.clamp(None, 5)) + 1e-6
        rdr = self.rate #final rate is rdr
        logdr = du #log of the modulation of prior rate
        return r-rdr+rdr*logdr
    



# class NegBinomial(nn.Module):
#     def __init__(self, 
#                  log_r, 
#                  logit_p, 
#                  t=0.0, 
#                  reparam_type = "gamma", 
#                  max_count = 15, 
#                  tau=1.0,
#                  latent_dim = 128):
#         super().__init__()
#         self.log_r = log_r # log dispersion
#         self.logit_p = logit_p # encoder modulation
#         self.r = torch.exp(
#             log_r.clamp(None, 5)
#         ) + 1e-6
#         self.p = torch.sigmoid(
#             logit_p.clamp(-5,5)
#         )
#         self.tau = tau 
#         self.max_count = max_count 
#         self.reparam_type = reparam_type
#         self.latent_dim = latent_dim

#         if self.reparam_type == "gamma":
#             # Convert to Gamma-Poisson
#             self.gamma_scale = (1-self.p) / self.p
#             self.rate = torch.distributions.Gamma(self.r, self.gamma_scale).rsample()
#             self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
#         elif self.reparam_type == "gumbel":
#             self.gumbel_module = GumbelNegBinomial(
#                 latent_dim=latent_dim, 
#                 max_count=max_count, 
#                 tau=tau
#             )
         
#         else:
#             raise ValueError(f"Unknown reparam_type: {reparam_type}")
        
#         self.t = t
#     def rsample(self, hard: bool = False):
#         if self.reparam_type == "gamma":
#             x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  # inter-event times
#             times = torch.cumsum(x, dim=0)  # arrival times of events
#             indicator = times < 1.0 # did events arrive before the end of the time interval
#             if not (hard or self.t == 0): # soften the indicator function
#                 indicator = torch.sigmoid((1.0 - times) / self.t)
#             return indicator.sum(0).float()
#         elif self.reparam_type == "gumbel":
#             #logits = self.logits(self.logit_p) # [B, max_count]
#             # logits = self.logits_layer(self.logit_p).view(-1, self.latent_dim, self.max_count)
#             # gumbels = -torch.empty_like(logits).exponential_().log()
#             # y = F.softmax((logits + gumbels) / self.tau, dim = -1)
#             # if hard:
#             #     index = y.max(dim=-1, keepdim=True)[1]
#             #     y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
#             #     y = (y_hard-y).detach() + y
#             # z = (y * self.count_range).sum(-1)
#             # return z 
#             return self.gumbel_module(self.logit_p, hard=hard)
            

#     def kl(self, log_r_prior, logit_p_prior):
#         r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
#         p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
#         q_p = self.p

#         ab = p * q_p
#         term = torch.log(q_p) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
#         return r * term
    



class NegBinomial(nn.Module):
    def __init__(self, 
                 reparam_type = "gamma", 
                 max_count = 15, 
                 tau=1.0,
                 latent_dim = 128):
        super().__init__()
        self.tau = tau 
        self.max_count = max_count 
        self.reparam_type = reparam_type
        self.latent_dim = latent_dim

        if self.reparam_type == "gumbel":
            self.gumbel_module = GumbelNegBinomial(
                latent_dim=latent_dim, 
                max_count=max_count, 
                tau=tau
            )
         
        else:
            raise ValueError(f"Unknown reparam_type: {reparam_type}")
        


    def rsample(self, log_r, logit_p, t= 0.0, hard: bool = False):
        r = torch.exp(log_r.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p.clamp(-5, 5))

        if self.reparam_type == "gamma":
            gamma_scale = (1 - p) / p
            rate = torch.distributions.Gamma(r, gamma_scale).rsample()
            n_trials = int(math.ceil(max(rate.max().item(), 1) * 5))
            x = torch.distributions.Exponential(rate).rsample((n_trials,))
            times = torch.cumsum(x, dim=0)
            indicator = times < 1.0
            if not (hard or t == 0):
                indicator = torch.sigmoid((1.0 - times) / t)
            z = indicator.sum(0).float()
            return z, None
    
        elif self.reparam_type == "gumbel":
            return self.gumbel_module(logit_p, hard=hard)
        
                
    def kl(self, log_r_prior, logit_p_prior, logit_p_post):
        r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p_prior.clamp(-5, 5))  # prior p
        q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))  # posterior p

        ab = p * q_p
        term = torch.log(q_p + 1e-8) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
        return r * term
    



class GumbelNegBinomial(nn.Module):
    def __init__(self, latent_dim=128, max_count=15, tau=1.0, init_lambda = 5.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_count = max_count
        self.tau = tau

        self.logits_layer = nn.Linear(latent_dim, latent_dim * max_count)
        nn.init.xavier_uniform_(self.logits_layer.weight)

        #============
        pmf = poisson.pmf(np.arange(max_count), mu=init_lambda)
        log_pmf = np.log(pmf + 1e-8)  # avoid log(0)
        tiled_log_pmf = np.tile(log_pmf, latent_dim)  # [D × K]

        self.logits_layer.bias.data = torch.tensor(tiled_log_pmf, dtype=torch.float)
        #===========



        self.logits_layer.bias.data.fill_(0.0)

        self.register_buffer("count_range", torch.arange(max_count).float())

    def forward(self, logit_p, hard=False):
        """
        Args:
            logit_p: Tensor of shape [B, latent_dim] from encoder
            hard: bool, whether to use hard (straight-through) sampling
        Returns:
            z: Tensor of shape [B, latent_dim], sampled latent count vector
        """
        B = logit_p.shape[0]

        logits = self.logits_layer(logit_p)  # [B, latent_dim * max_count]
        logits = logits.view(B, self.latent_dim, self.max_count)  # [B, D, K]

        gumbels = -torch.empty_like(logits).exponential_().log()  # [B, D, K]
        y = F.softmax((logits + gumbels) / self.tau, dim=-1)      # [B, D, K]
        # print(y[0, 0])

        if hard:
            index = y.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
            y = (y_hard - y).detach() + y  # Straight-through

        count_range = self.count_range.to(y.device)
        z = (y * count_range).sum(-1)  # [B, latent_dim], expected count per dim
        return z

    

