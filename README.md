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

## Hierarchical NB-VAE

`--model_type hnb` enables the multi-scale HNB-VAE architecture. It adds a
bottom-up feature hierarchy and a top-down sequence of
spatial negative-binomial latent groups. The first group has a learned global
prior; every later group has a prior conditioned on the accumulated top-down
state. Posterior parameters use neutral residual multipliers, so `q=p` at
initialization.

Run the MNIST CCH configuration with:

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
  --kl cch \
  --hnb_groups_per_scale 4,4 \
  --hnb_channels_per_scale 64,32 \
  --hnb_latent_channels_per_scale 16,16 \
  --bsize 64 \
  --max_epochs 300
```

The HNB objective supports:

- `--kl gamma`: Gamma-overlapping latent variables and exact Gamma KL;
- `--kl cch`: Gamma/CTS samples and the conjugate correction, reusing the
  same CTS sample in the decoder path and KL correction;
- `--kl mc`: the original relaxed-NB Monte Carlo baseline.

The one-layer `--kl analytical` formula assumes shared dispersion and is
therefore deliberately rejected for HNB, whose residual posterior changes
both NB parameters. HNB training also supports scale-normalized group weights
(`--hnb_kl_balance scale`), per-group free bits (`--hnb_free_bits`), and
top-to-bottom ladder warm-up using `trainer.schedule_epochs`.

By default the latent spatial sizes are derived from the input resolution. Use
`--hnb_spatial_sizes 7,14` to set them explicitly; every HNB list must have the
same number of scale entries.
