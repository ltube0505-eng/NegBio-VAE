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

## CCH-VAE

The conjugate-corrected hybrid objective is available for the negative-
binomial model with Gamma/CTS reparameterization:

```bash
python dis_main.py \
  --config_path configs/cchconfig.yaml \
  --dataset MNIST \
  --model_type negbio \
  --reparam_type gamma \
  --kl cch \
  --cch_tau 0.1 \
  --cts_max_count 64
```

`--cch_detach_phi=False` (the default) keeps the full pathwise gradient
through the CTS sample reused by the reconstruction and KL correction. Set it
to `True` for the lower-variance, biased stop-gradient ablation. CCH requires a
low CTS temperature; the recommended range is 0.05--0.2. Its two-parameter
posterior also changes the encoder output size, so checkpoints trained with a
one-parameter dispersion-sharing encoder are not shape-compatible.

Training logs `cch_phi_ratio`, `cch_clamp_rate`, and
`cch_cts_truncation_rate`. Increase `--cts_max_count` when the truncation rate
is not close to zero.
