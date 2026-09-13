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

## Hierarchical NB-VAE

`--model_type hnb` enables the multi-scale HNB-VAE architecture. It adds a
bottom-up feature hierarchy and a top-down sequence of
spatial negative-binomial latent groups. The first group has a learned global
prior; every later group has a prior conditioned on the accumulated top-down
state. Posterior parameters use neutral residual multipliers, so `q=p` at
initialization.

Run the MNIST MC-KL configuration with:

```bash
bash scripts/train_hnb.sh
```

or configure it directly:

```bash
python dis_main.py \
  --config_path configs/hnbconfig.yaml \
  --dataset MNIST \
  --model_type hnb \
  --reparam_type gamma \
  --kl mc \
  --hnb_groups_per_scale 4,4 \
  --hnb_channels_per_scale 64,32 \
  --hnb_latent_channels_per_scale 16,16 \
  --bsize 64 \
  --max_epochs 300
```

HNB uses the relaxed-NB Monte Carlo KL implementation (`--kl mc`). At every
latent group it evaluates both `log q(z | x)` with the posterior dispersion
and `log p(z)` with that group's conditional-prior dispersion. The one-layer
`--kl analytical` formula assumes shared dispersion and is therefore
deliberately rejected for HNB, whose residual posterior changes both NB
parameters. The original one-layer model still supports its analytical
shared-dispersion objective as well as `--kl mc`.

HNB training also supports scale-normalized group weights
(`--hnb_kl_balance scale`), per-group free bits (`--hnb_free_bits`), and
top-to-bottom ladder warm-up using `trainer.schedule_epochs`.

For unconditional generation, `HNBVAE.sample_prior(num_samples)` skips the
bottom-up encoder and samples each latent group from its top-down conditional
prior before decoding the final state.

By default the latent spatial sizes are derived from the input resolution. Use
`--hnb_spatial_sizes 7,14` to set them explicitly; every HNB list must have the
same number of scale entries.
