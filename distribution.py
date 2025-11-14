import math
import os

import numpy as np
from numpy.ma.core import zeros
import pytorch_lightning as pl
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import poisson
from torch.distributions import Laplace as TorchLaplace
from torch.distributions import kl_divergence
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class Poisson:
    # This Poisson Class is copied from Poisson-VAE https://github.com/hadivafaii/PoissonVAE
    def __init__(self, log_rate, t=0.0):
        self.log_rate = log_rate
        self.rate = torch.exp(
            self.log_rate.clamp(None, 5)
        ) + 1e-6
        self.n_trials = int(math.ceil(max(self.rate.max().item(),1)*5))
        self.t = t

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def rsample(self, hard: bool = False):
        x = torch.distributions.Exponential(self.rate).rsample((self.n_trials,))  
        times = torch.cumsum(x, dim=0) 
        indicator = times < 1.0 
        if not (hard or self.t == 0): 
            indicator = torch.sigmoid((1.0 - times) / self.t)
        return indicator.sum(0).float()
    def kl(self, prior, du):
        r = torch.exp(prior.clamp(None, 5)) + 1e-6
        rdr = self.rate
        logdr = du 
        return r-rdr+rdr*logdr



class GammaSampler:
    def __init__(self, 
                 t=0.0,
                 num_samples = 5):
        self.t = t
        self._cached_log_r = None
        self._cached_logit_p = None
        self.num_samples = num_samples

    def __call__(self, 
                 log_r,
                 logit_p, 
                 hard=False):
        self._cached_log_r = log_r
        self._cached_logit_p = logit_p
        return self.rsample(log_r, logit_p, hard)

    @property
    def mean(self):
        return self.r * (1 - self.p) / self.p

    @property
    def variance(self):
        return self.r * (1 - self.p) / (self.p ** 2)


    def rsample(self, log_r, logit_p, hard=False):
        self.r = torch.exp(log_r.clamp(None, 5)) + 1e-6
        self.p = torch.sigmoid(logit_p.clamp(-5, 5))

        gamma_rate = self.p / (1 - self.p + 1e-8)                  
        lam = torch.distributions.Gamma(self.r, gamma_rate).rsample() + 1e-6

        n_trials = min(int(math.ceil(max(lam.max().item(), 1) * 5)), 826)
        x = torch.distributions.Exponential(lam).rsample((n_trials,))

        times = torch.cumsum(x, dim=0)
        indicator = times < 1.0
        if not (hard or self.t == 0):
            indicator = torch.sigmoid((1.0 - times) / self.t)
        z = indicator.sum(0).float()
        return z

    def kl_mc(self, log_r_prior, logit_p_prior, num_samples=5):

        log_r_post = self._cached_log_r
        logit_p_post = self._cached_logit_p

        r_q = torch.exp(log_r_post.clamp(None, 5)) + 1e-6
        
        p_q = torch.sigmoid(logit_p_post.clamp(-5, 5))

        rate_q = p_q / (1 - p_q + 1e-8)

        q_dist = torch.distributions.Gamma(r_q, rate_q)

        r_p = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p_p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
    

        rate_p = p_p / (1 - p_p + 1e-8)                              
        p_dist = torch.distributions.Gamma(r_p, rate_p)

        samples = q_dist.rsample((num_samples,))
        kl = (q_dist.log_prob(samples) - p_dist.log_prob(samples)).mean(dim=0)
        return kl


class GumbelSampler(nn.Module):
    def __init__(self, 
                 max_count=15, 
                 tau=1.0,
                 num_samples = 5):
        super().__init__()
        self.max_count = max_count
        self.tau = tau
        self.register_buffer("count_range", torch.arange(max_count).float())
        self._cached_log_r = None
        self._cached_logit_p = None
        self.num_samples = num_samples

    def forward(self, log_r, logit_p, hard=False):
        self._cached_log_r = log_r
        self._cached_logit_p = logit_p

        r = torch.exp(log_r.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p.clamp(-5, 5))

        gamma_rate = p / (1 - p + 1e-8)                             
        rate = torch.distributions.Gamma(r, gamma_rate).rsample()

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
    
    def kl_mc(self, log_r_prior, logit_p_prior):

        log_r_post = self._cached_log_r
        logit_p_post = self._cached_logit_p

        r_q = torch.exp(log_r_post.clamp(None, 5)) + 1e-6
        p_q = torch.sigmoid(logit_p_post.clamp(-5, 5))

        rate_q = p_q /(1 - p_q + 1e-8)

        q_dist = torch.distributions.Gamma(r_q, rate_q)

        r_p = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p_p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
        
        rate_p = p_p / (1 - p_p + 1e-8)                              

        p_dist = torch.distributions.Gamma(r_p, rate_p)

        samples = q_dist.rsample((self.num_samples,))
        kl = (q_dist.log_prob(samples) - p_dist.log_prob(samples)).mean(dim=0)
        return kl


class NegBinomial(nn.Module):
    def __init__(self, 
                 reparam_type="gamma", 
                 max_count=15, 
                 tau=1.0,
                 num_samples = 5):
        super().__init__()
        self.reparam_type = reparam_type
        self.num_samples = num_samples
        if reparam_type == "gamma":
            self.strategy = GammaSampler(t=tau, num_samples=self.num_samples)
        elif reparam_type == "gumbel":
            self.strategy = GumbelSampler(max_count=max_count, tau=tau, num_samples=self.num_samples)
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
            self.strategy.t = t  
        return self.strategy(log_r, logit_p, hard=hard)

    def kl(self, log_r_prior, logit_p_prior, logit_p_post):
        r = torch.exp(log_r_prior.clamp(None, 5)) + 1e-6
        p = torch.sigmoid(logit_p_prior.clamp(-5, 5))
        q_p = torch.sigmoid(logit_p_post.clamp(-5, 5))
        a = p
        b = q_p / (p + 1e-8)
        ab = q_p
        term = torch.log(b) + (1 - ab)/(ab + 1e-8) * torch.log((1 - ab + 1e-8) / (1 - a + 1e-8))
        return r * term

    def kl_mc(self, log_r_prior, logit_p_prior):
        if self.reparam_type == "gamma":
            return self.strategy.kl_mc(log_r_prior, logit_p_prior)
        elif self.reparam_type == "gumbel":
            return self.strategy.kl_mc(log_r_prior, logit_p_prior)
        else:
            raise NotImplementedError(f"No KL_MC implemented for reparam_type: {self.reparam_type}")
    

class Categorical(RelaxedOneHotCategorical):
    def __init__(self, logits, temp=1.0):
        temp = max(temp, torch.finfo(torch.float).eps)
        logits = logits.clamp(-10, 10)  
        super().__init__(temperature=temp, logits=logits)
        self._logits = logits
        self._probs = F.softmax(logits, dim=-1)
        
    def comput_kl(self, prior_logits = None):

        p_dist = dists.Categorical(probs=self._probs)

        if prior_logits is None:
            q_probs = torch.full_like(self._probs, fill_value=1.0 / self._probs.size(-1))
            q_dist = dists.Categorical(probs=q_probs)
        else:
            prior_logits = prior_logits.clamp(-10, 10)
            q_probs = F.softmax(prior_logits, dim=-1)
            q_dist = dists.Categorical(probs=q_probs)
    
        return dists.kl.kl_divergence(p_dist, q_dist)
    
    def rsample(self, hard=False):
        y = super().rsample()  
        if hard:
            # Straight-through estimator
            index = y.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
            y = (y_hard - y).detach() + y
        return y



class Laplace:
    def __init__(self, loc, log_scale, t=1.0, clamp=5.0):
        self.t = t
        self.loc = loc
        self.log_scale = log_scale.clamp(-clamp, clamp)
        self.scale = torch.exp(self.log_scale).clamp(min=1e-6) * self.t
        self.dist = TorchLaplace(self.loc, self.scale)

    @property
    def mean(self):
        return self.dist.mean

    @property
    def variance(self):
        return self.dist.variance

    def rsample(self, hard: bool = False):
        return self.dist.rsample()

    def kl(self, prior=None):
        if prior is None:
            prior = TorchLaplace(
                loc=torch.zeros_like(self.loc), 
                scale=torch.ones_like(self.scale)
            )
        return kl_divergence(self.dist, prior)
    

class Gaussian:
    def __init__(self, loc, log_scale, t=1.0, clamp=5.0):
        self.t = t
        self.loc = loc
        self.log_scale = log_scale.clamp(-clamp, clamp)
        self.scale = torch.exp(self.log_scale).clamp(min=1e-6) * self.t
        self.dist = dists.Normal(self.loc, self.scale)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale.pow(2)

    def rsample(self):
        return self.dist.rsample()

    def kl(self, prior=None):
        if prior is None:
            prior = dists.Normal(
                loc=torch.zeros_like(self.loc),
                scale=torch.ones_like(self.scale)
            )
        else:
            prior_loc, prior_scale = prior
            prior = dists.Normal(prior_loc, prior_scale.clamp(min=1e-6))

        return dists.kl.kl_divergence(self.dist, prior)
