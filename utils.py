from typing import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from torchvision.models import inception_v3

import wandb as wdb
from architecture import *


def build_encoder(cfg):
    encoder_cfg = cfg['encoder']
    name = encoder_cfg['type'].lower()
    latent_dim = encoder_cfg.get('latent_dim', 128)
    if name == 'linear':
        print(encoder_cfg['input_dim'][cfg['datasetname']])
        return LinearEncoder(
            input_dim=encoder_cfg['input_dim'][cfg['datasetname']],
            latent_dim=latent_dim
        )
    
    elif name == 'conv':
        return ConvEncoder(latent_dim=latent_dim,
                           dataset = cfg['datasetname'],
                           use_norm=encoder_cfg['use_norm'],
                           bias = encoder_cfg['bias'])
        # return ConvEncoder(encoder_cfg,
        #                    cfg['dataset']['name'], 
        #                    latent_dim=128, 
        #                    n_ch=32)
    elif name == 'mlp':
          return MLPEncoder(
                input_dim=encoder_cfg['input_dim'][cfg['datasetname']],
                latent_dim=latent_dim,
                expand=encoder_cfg.get('expand', 32),
                normalize=encoder_cfg.get('normalize', False),
                bias=encoder_cfg.get('bias', False),
          )
    else:
        raise ValueError(f"Unsupported encoder type: {name}")






def build_decoder(cfg):
    decoder_cfg = cfg['decoder']
    name = decoder_cfg['type'].lower()
    latent_dim = decoder_cfg.get('latent_dim', 128)
    if name == 'linear':
        return LinearDecoder(
            latent_dim=latent_dim,
            output_dim=cfg['encoder']['input_dim'][cfg['datasetname']]
        )
    elif name == 'conv':
        return ConvDecoder(latent_dim=latent_dim, 
                           out_channels=cfg['decoder']['out_channel'][cfg['datasetname']],
                           size=cfg['decoder']['size'][cfg['datasetname']]
        )
    elif name == 'mlp':
          return MLPDecoder(
                latent_dim=latent_dim,
                output_dim=cfg['encoder']['input_dim'][cfg['datasetname']],
                normalize=decoder_cfg.get('normalize', False),
                bias=decoder_cfg.get('bias', False),
                activation_fn=decoder_cfg.get('activation_fn', 'swish'),
          )
    else:
        raise ValueError(f"Unsupported decoder type: {name}")
    

def log_latent_mean_vs_var(logger, 
                           z, 
                           save_dir, 
                           step_name = "val", 
                           caption = "Latent mean vs variance",
                           savelocal = True):
    if isinstance(z, torch.Tensor):
        z_mean = z.mean(dim=0).cpu()
        z_var = z.var(dim=0).cpu()
    else:
        z_mean = np.mean(z, axis=0)
        z_var = np.var(z, axis=0)


    # z_mean = z.mean(dim=0).cpu()
    # z_var = z.var(dim=0).cpu()

    # Plot mean vs var
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(z_mean, z_var, alpha=0.6, label='Latent units')
    ax.plot([0, z_mean.max()], [0, z_mean.max()], 'r--', label='Poisson (mean=var)')
    ax.set_xlabel('Mean of $z_i$')
    ax.set_ylabel('Variance of $z_i$')
    ax.set_title('Latent Mean vs Variance')
    ax.legend()
    plt.tight_layout()
    if savelocal is True:
        mean_var_path = os.path.join(save_dir, "mean_var.pdf")
        fig.savefig(mean_var_path)
    logger.log({
        "latent_mean_vs_var": wdb.Image(fig, caption=caption),
    })
    plt.close(fig)


def get_overdispersion_index(z, eps=1e-8):
    if isinstance(z, torch.Tensor):
        z_mean = z.mean(dim=0).cpu()
        z_var = z.var(dim=0).cpu()
    else:
        z_mean = np.mean(z, axis=0)
        z_var = np.var(z, axis=0)

    # z_mean = z.mean(dim=0).cpu()
    # z_var = z.var(dim=0).cpu()

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




def plot_fano(z, save_dir, logger):
    if isinstance(z, torch.Tensor):
        z_mean = z.mean(dim=0).cpu()
        z_var = z.var(dim=0).cpu()
    else:
        z_mean = np.mean(z, axis=0)
        z_var = np.var(z, axis=0)


    fano_factors = z_var / (z_mean + 1e-8)

    fano_fig = plt.figure(figsize=(6, 4))
    sns.histplot(fano_factors, bins=20, kde=True, color='skyblue')
    plt.axvline(1.0, color='red', linestyle='--', label='Poisson baseline (Fano=1)')
    plt.title('Fano Factor Distribution across Latent Units')
    plt.xlabel('Fano Factor')
    plt.ylabel('Number of Latent Units')
    plt.legend()
    plt.tight_layout()
    fano_path = os.path.join(save_dir, "fano_factor_hist.pdf")
    fano_fig.savefig(fano_path)
    plt.close(fano_fig)

    logger.experiment.log({
        "fano_factor_hist": wdb.Image(fano_fig, caption = "Fano Factor Distribution across Latent Units")
    })
    



def spike_count_hist(z, save_dir, logger, top_k = 3):
    # Select top-k latent units by variance
    if isinstance(z, torch.Tensor):
        z_mean = z.mean(dim=0).cpu()
        z_var = z.var(dim=0).cpu()
    else:
        z_mean = np.mean(z, axis=0)
        z_var = np.var(z, axis=0)

    selected_units = np.argsort(z_var)[-top_k:][::-1]  # descending order
    selected_units = [0, 10, 25] if z.shape[1] >= 26 else list(range(min(3, z.shape[1])))

    spike_fig, axs = plt.subplots(1, len(selected_units), figsize=(12, 4))
    for i, idx in enumerate(selected_units):
        unit_counts = z[:, idx]
        sns.histplot(unit_counts, bins=range(0, int(unit_counts.max()) + 2),
                     stat='probability', kde=False, ax=axs[i],
                     color='steelblue', edgecolor='black')
        axs[i].set_title(f'Latent Unit {idx}')
        axs[i].set_xlabel('Spike Count')
        axs[i].set_ylabel('Probability')
        axs[i].set_xlim(left=0)
    plt.suptitle('Spike Count Distributions for Selected Latent Units')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    spike_hist_path = os.path.join(save_dir, "spike_count_histograms.pdf")
    spike_fig.savefig(spike_hist_path)
    plt.close(spike_fig)

    logger.experiment.log({
        "spike_count_histograms": wdb.Image(spike_fig)
    })



def log_dead_neurons_diagnostics(kl_diag, dead_mask, logger, step_name="val"):
    fig, ax = plt.subplots(figsize=(6, 4))
    dims = np.arange(len(kl_diag))
    kl_vals = kl_diag
    ax.bar(dims[dead_mask], kl_vals[dead_mask], color="red", label="Dead neuron", alpha=0.5)
    ax.bar(dims, kl_vals, color="blue", label="KL per dim")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"[{step_name}] KL per latent dim")
    ax.legend()
    plt.tight_layout()

    logger.experiment.log({
        f"kl_per_dim": wdb.Image(fig, caption="KL per latent dim (red = dead)"),
    })
    plt.close(fig)




def compute_inception_score(images, batch_size=32, splits=10, device = "cuda"):
    """
    images: Tensor of shape [N, 3, H, W] and in range [0, 1]
    """
    model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
    
    def get_pred(x):
        with torch.no_grad():
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = model(x)
            return F.softmax(x, dim=1).cpu().numpy()

    N = images.shape[0]
    preds = np.zeros((N, 1000))
    for i in range(0, N, batch_size):
        batch = images[i:i + batch_size].to(device)
        preds[i:i + batch_size] = get_pred(batch)

    scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits)]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-6) - np.log(py + 1e-6))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    
    return np.mean(scores), np.std(scores)


