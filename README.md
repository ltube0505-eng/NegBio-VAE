# Negative Binomial Variational Autoencoders for Overdispersed Latent Modeling

Official implementation for the paper "Negative Binomial Variational Autoencoders for Overdispersed Latent Modeling"(CVPR 2026).

## Overview

We propose NegBio-VAE, a novel variational autoencoder that leverages the Negative Binomial (NB) distribution to model overdispersed discrete latent variables, inspired by neural spike trains in biological systems.

**Key contributions:**
- NegBio-VAE models overdispersed latent spike counts via a dispersion parameter, enabling more flexible latent representations.
- Efficient training strategies combining tailored KL estimators and differentiable reparameterizations ensure stable optimization.
- Strong empirical performance across four benchmarks, improving reconstruction, generation, and downstream representation quality.

## Environment

```bash
conda env create -f nbvae.yml
```

## Training

To train a model, run
```bash
cd scripts/
./train.sh
```
