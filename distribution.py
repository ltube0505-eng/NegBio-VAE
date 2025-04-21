import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning.loggers import wandb
from scipy.stats import poisson
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets

import wandb as wdb


class Poisson:
    # This code is copied from Poisson-VAE
    def __init__(self, log_rate, t=0.0):
        self.log_rate = log_rate
        self.rate = torch.exp(
            self.log_rate.clamp(None, 5)
        ) + 1e-6
        self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
        self.t = t

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate
    
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
    
    def linear_decoder_exact_recon_loss(self, x, phi):
        mean = self.rate
        var = self.rate

        a = phi.pow(2).sum(0)

        mse = x - mean @ phi.T
        mse = mse.pow(2).sum(1)
        recon_loss = mse + var @ a

        return recon_loss.mean()


class GammaSampler:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self, log_r, logit_p, hard=False):
        return self.rsample(log_r, logit_p, hard)
    
    @property
    def mean(self):
        return self.r * (1 - self.p) / self.p

    @property
    def variance(self):
        return self.r * (1 - self.p) / (self.p ** 2)

    def rsample(self, log_r, logit_p, hard=False):
        r = torch.exp(log_r.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p.clamp(-5, 5))

        self.r = r
        self.p = p

        gamma_scale = (1 - p) / p
        rate = torch.distributions.Gamma(r, gamma_scale).rsample() + 1e-6
        n_trials = min(int(math.ceil(max(rate.max().item(), 1) * 5)), 826)
        # n_trials = 826
        x = torch.distributions.Exponential(rate).rsample((n_trials,))
        times = torch.cumsum(x, dim=0)
        indicator = times < 1.0
        if not (hard or self.t == 0):
            indicator = torch.sigmoid((1.0 - times) / self.t)
        z = indicator.sum(0).float()
        return z
    
    def linear_decoder_exact_recon_loss(self, x, phi, alpha = 0.1, eps=1e-6):
        # mean_z = self.r * (1 - self.p) / self.p
        # mu = mean_z @ phi.T
        mu = (self.mean @ phi.T).clamp_min(eps) 
        r = 1.0 / alpha
        log_prob = (
            torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
            + r * torch.log(r / (r + mu))
            + x * torch.log(mu / (r + mu))
        )

        return -log_prob.sum(dim=1).mean()
    

    def kl_mc(self, log_r_prior, logit_p_prior, num_samples=1):
        r_q, p_q = self.r, self.p
        r_p = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p_p = torch.sigmoid(logit_p_prior.clamp(-5, 5))

        rate_q = (1 - p_q) / p_q
        rate_p = (1 - p_p) / p_p

        q_dist = torch.distributions.Gamma(r_q, rate_q)
        p_dist = torch.distributions.Gamma(r_p, rate_p)

        samples = q_dist.rsample((num_samples,))  # [S, B, L]
        log_q = q_dist.log_prob(samples)
        log_p = p_dist.log_prob(samples)

        kl = (log_q - log_p).mean(dim=0).sum(dim=-1).mean()
        return kl


class GumbelSampler(nn.Module):
    def __init__(self, max_count=15, tau=1.0):
        super().__init__()
        self.max_count = max_count
        self.tau = tau
        self.register_buffer("count_range", torch.arange(max_count).float())

    def forward(self, log_r, logit_p, hard=False):
        r = torch.exp(log_r.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p.clamp(-5, 5))
        gamma_scale = (1 - p) / p
        rate = torch.distributions.Gamma(r, gamma_scale).rsample()

        k = self.count_range.view(1, 1, -1)
        rate = rate.unsqueeze(-1)
        log_pmf = k * rate.log() - rate - torch.lgamma(k + 1)

        gumbel = -torch.empty_like(log_pmf).exponential_().log()
        y = F.softmax((log_pmf + gumbel) / self.tau, dim=-1)

        if hard:
            index = y.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
            y = (y_hard - y).detach() + y

        z = (y * self.count_range.to(rate.device)).sum(-1)
        return z
    
    def kl_mc(self, log_r_prior, logit_p_prior, logit_p_post, num_samples=1):
        # Construct approximate categorical distributions from softmax
        r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
        q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))

        # For Gamma-derived prior rate
        gamma_rate = (1 - p) / p
        rate = torch.distributions.Gamma(r, gamma_rate).rsample().unsqueeze(-1)
        k = self.count_range.view(1, 1, -1)
        log_pmf_prior = k * rate.log() - rate - torch.lgamma(k + 1)

        # Soft categorical from q
        # forward() returns y [B, L, max_count]
        with torch.no_grad():
            y = self.forward(log_r_prior, logit_p_post, hard=False)  # softmax logits
            probs = F.softmax(y.unsqueeze(-1) * self.count_range.to(y.device), dim=-1)  # [B, max_count]

        # Soft KL: q * (log q - log p)
        kl = (probs * (probs.log() - log_pmf_prior)).sum(dim=-1).mean()
        return kl


class NegBinomial(nn.Module):
    def __init__(self, reparam_type="gamma", max_count=15, tau=1.0):
        super().__init__()
        self.reparam_type = reparam_type
        if reparam_type == "gamma":
            self.strategy = GammaSampler(t=tau)
        elif reparam_type == "gumbel":
            self.strategy = GumbelSampler(max_count=max_count, tau=tau)
        else:
            raise ValueError(f"Unsupported reparam_type: {reparam_type}")
        
    @property
    def mean(self):
        return self.strategy.mean

    @property
    def variance(self):
        return self.strategy.variance

    def rsample(self, log_r, logit_p, t=0.0, hard=False):
        if self.reparam_type == "gamma":
            self.strategy.t = t  # update t on the fly
        return self.strategy(log_r, logit_p, hard=hard)

    def kl(self, log_r_prior, logit_p_prior, logit_p_post):
        r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
        q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))

        ab = p * q_p
        term = torch.log(q_p + 1e-8) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
        return r * term
    
    def kl_mc(self, log_r_prior, logit_p_prior, logit_p_post, num_samples=1):
        if self.reparam_type == "gamma":
            return self.strategy.kl_mc(log_r_prior, logit_p_prior, num_samples=num_samples)
        elif self.reparam_type == "gumbel":
            return self.strategy.kl_mc(log_r_prior, logit_p_prior, logit_p_post, num_samples=num_samples)
        else:
            raise NotImplementedError(f"No KL_MC implemented for reparam_type: {self.reparam_type}")
    
    def linear_decoder_exact_recon_loss(self, *args, **kwargs):
        return self.strategy.linear_decoder_exact_recon_loss(*args, **kwargs)

# class NegBinomial_Gamma:
#     def __init__(self, log_rate, logit_p, t=0.0):
#         self.log_rate = log_rate
#         self.rate = torch.exp(
#             self.log_rate.clamp(None, 5)
#         ) + 1e-6
#         self.p = torch.sigmoid(logit_p.clamp(-5, 5))
#         self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
#         self.t = t


#     def rsample(self, hard: bool = False):
#         gamma_scale = (1 - self.p) / self.p
#         rate = torch.distributions.Gamma(self.rate, gamma_scale).rsample()
#         x = torch.distributions.Exponential(rate).rsample((self.n_trials,))
#         times = torch.cumsum(x, dim=0)
#         indicator = times < 1.0
#         if not (hard or self.t == 0):
#             indicator = torch.sigmoid((1.0 - times) / self.t)
#         z = indicator.sum(0).float()
#         return z
#     def kl(self, log_r_prior, logit_p_prior, logit_p_post):
#         r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
#         p = torch.sigmoid(logit_p_prior.clamp(-5, 5))  # prior p
#         q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))  # posterior p

#         ab = p * q_p
#         term = torch.log(q_p + 1e-8) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
#         return r * term
    

# class NegBinomial(nn.Module):
#     def __init__(self, 
#                  reparam_type = "gamma", 
#                  max_count = 15, 
#                  tau=1.0,):
#         super().__init__()
#         self.tau = tau 
#         self.max_count = max_count 
#         self.reparam_type = reparam_type
    
#         if self.reparam_type == "gumbel":
#             self.gumbel_module = GumbelNegBinomial(
#                 max_count=max_count, 
#                 tau=tau
#             )
         
#         elif reparam_type != "gamma":
#             raise ValueError(f"Unknown reparam_type: {reparam_type}")
        


#     def rsample(self, log_r, logit_p, t= 0.0, hard: bool = False):
#         r = torch.exp(log_r.clamp(None, 5)) + 1e-6
#         p = torch.sigmoid(logit_p.clamp(-5, 5))

#         if self.reparam_type == "gamma":
#             gamma_scale = (1 - p) / p
#             rate = torch.distributions.Gamma(r, gamma_scale).rsample()
#             n_trials = int(math.ceil(max(rate.max().item(), 1) * 5))
#             x = torch.distributions.Exponential(rate).rsample((n_trials,))
#             times = torch.cumsum(x, dim=0)
#             indicator = times < 1.0
#             if not (hard or t == 0):
#                 indicator = torch.sigmoid((1.0 - times) / t)
#             z = indicator.sum(0).float()
#             return z
    
#         elif self.reparam_type == "gumbel":
#             gamma_scale = (1 - p) / p
#             rate = torch.distributions.Gamma(r, gamma_scale).rsample()
#             z = self.gumbel_module(rate, hard=hard)
#             return z
        
                
#     def kl(self, log_r_prior, logit_p_prior, logit_p_post):
#         r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
#         p = torch.sigmoid(logit_p_prior.clamp(-5, 5))  # prior p
#         q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))  # posterior p

#         ab = p * q_p
#         term = torch.log(q_p + 1e-8) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
#         return r * term
    


# class GumbelNegBinomial(nn.Module):
#     def __init__(self, 
#                  max_count=15, 
#                  tau=1.0):
#         super().__init__()
#         self.max_count = max_count
#         self.tau = tau
#         self.register_buffer("count_range", torch.arange(max_count).float())

#     def forward(self, rate, hard = False):
#         k = self.count_range.view(1, 1, -1)  # [1,1,K]
#         rate = rate.unsqueeze(-1)  # [B, D, 1]
#         log_pmf = k * rate.log() - rate - torch.lgamma(k + 1)  # [B, D, K]

#         # Gumbel noise
#         gumbel = -torch.empty_like(log_pmf).exponential_().log()
#         y = F.softmax((log_pmf + gumbel) / self.tau, dim=-1)

#         if hard:
#             index = y.max(dim=-1, keepdim=True)[1]
#             y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
#             y = (y_hard - y).detach() + y

#         z = (y * self.count_range.to(rate.device)).sum(-1)  # [B, D]
#         return z


    

