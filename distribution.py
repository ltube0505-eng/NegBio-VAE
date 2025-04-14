import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance




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
    



class NegBinomial:
    def __init__(self, log_r, logit_p, t=0.0):
        self.log_r = log_r # log dispersion
        self.logit_p = logit_p # encoder modulation
        self.r = torch.exp(
            log_r.clamp(None, 5)
        ) + 1e-6
        self.p = torch.sigmoid(
            logit_p.clamp(-5,5)
        )

        # Convert to Gamma-Poisson
        self.gamma_scale = (1-self.p) / self.p
        self.rate = torch.distributions.Gamma(self.r, self.gamma_scale).rsample()
        self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5)) # a large enough number of trials to sample from
        self.t = t
    def rsample(self, hard: bool = False):
        x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  # inter-event times
        times = torch.cumsum(x, dim=0)  # arrival times of events
        indicator = times < 1.0 # did events arrive before the end of the time interval
        if not (hard or self.t == 0): # soften the indicator function
            indicator = torch.sigmoid((1.0 - times) / self.t)
        return indicator.sum(0).float()
    def kl(self, log_r_prior, logit_p_prior):
        r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
        q_p = self.p

        ab = p * q_p
        term = torch.log(q_p) + (1 - ab) / (ab + 1e-8) * torch.log((1 - ab + 1e-8)/(1 - p + 1e-8))
        return r * term
    


    

