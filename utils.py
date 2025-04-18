from typing import *

import matplotlib.pyplot as plt
import numpy as np

from architecture import *


def build_encoder(cfg):
    encoder_cfg = cfg['encoder']
    name = encoder_cfg['type'].lower()
    latent_dim = encoder_cfg.get('latent_dim', 128)
    if name == 'linear':
        return LinearEncoder(
            input_dim=encoder_cfg.get('input_dim', 784),
            latent_dim=latent_dim
        )
    elif name == 'conv':
        return ConvEncoder(latent_dim=latent_dim,
                           dataset = cfg['dataset']['name'],
                           use_norm=encoder_cfg['use_norm'],
                           bias = encoder_cfg['bias'])
        # return ConvEncoder(encoder_cfg,
        #                    cfg['dataset']['name'], 
        #                    latent_dim=128, 
        #                    n_ch=32)
    else:
        raise ValueError(f"Unsupported encoder type: {name}")






def build_decoder(cfg):
    decoder_cfg = cfg['decoder']
    name = decoder_cfg['type'].lower()
    latent_dim = decoder_cfg.get('latent_dim', 128)
    if name == 'linear':
        return LinearDecoder(
            latent_dim=latent_dim,
            output_dim=decoder_cfg.get('output_dim', 784)
        )
    elif name == 'conv':
        return ConvDecoder(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported decoder type: {name}")
    

def log_latent_mean_vs_var(logger, z, step_name = "val", caption = "Latent mean vs variance"):
    z_mean = z.mean(dim=0).cpu()
    z_var = z.var(dim=0).cpu()

    # Plot mean vs var
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(z_mean, z_var, alpha=0.6, label='Latent units')
    ax.plot([0, z_mean.max()], [0, z_mean.max()], 'r--', label='Poisson (mean=var)')
    ax.set_xlabel('Mean of $z_i$')
    ax.set_ylabel('Variance of $z_i$')
    ax.set_title(f'[{step_name}] Latent Mean vs Variance')
    ax.legend()
    plt.tight_layout()

    logger.log({
        "latent_mean_vs_var": wdb.Image(fig, caption=caption),
    })
    plt.close(fig)


def get_overdispersion_index(z, eps=1e-8):
    z_mean = z.mean(dim=0).cpu()
    z_var = z.var(dim=0).cpu()

    overdispersion_index = ((z_var + eps) / (z_mean + eps)).mean().item()

    return overdispersion_index


def gumbel_entropy(y):
    # y: [B, D, K] softmax output
    return -(y * y.clamp(min=1e-8).log()).sum(dim=-1).mean()



def tonp(x: Union[torch.Tensor, np.ndarray]):
	if isinstance(x, np.ndarray):
		return x
	elif isinstance(x, torch.Tensor):
		return x.data.cpu().numpy()
	else:
		raise ValueError(type(x).__name__)
     

def find_last_contiguous_zeros(mask: np.ndarray, w: int):
	# mask = hist > 0.0
	m = mask.astype(bool)
	zero_count = 0
	for idx, val in enumerate(m[::-1]):
		if val == 0:
			zero_count += 1
		else:
			zero_count = 0

		if zero_count == w:
			return len(m) - (idx - w + 2)
	return 0


def find_critical_ids(mask: np.ndarray):
	# mask = hist > 0.0
	m = mask.astype(bool)

	first_zero = 0
	for i in range(1, len(m)):
		if m[i-1] and not m[i]:
			first_zero = i
			break

	last_zero = -1
	for i in range(len(m) - 2, -1, -1):
		if not m[i] and m[i+1]:
			last_zero = i
			break

	return first_zero, last_zero



