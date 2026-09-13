from typing import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchmetrics.image.inception import InceptionScore
from torchvision.models import inception_v3
import wandb as wdb
from architecture import *


def build_encoder(cfg):
    encoder_cfg = cfg['encoder']
    name = encoder_cfg['type'].lower()
    dist_type = cfg['model']['name']
    kl_type = cfg['model'].get('kl', 'analytical')
    base_latent_dim = encoder_cfg.get('latent_dim', 128)
    latent_dim = (
            base_latent_dim * 2 
            if dist_type in ['laplace', 'gaussian']
            or (dist_type == 'negbio' and kl_type == 'mc')
            else base_latent_dim
        )

    if name == 'linear':
        return LinearEncoder(
            input_dim=encoder_cfg['input_dim'][cfg['dataset']['name']],
            latent_dim=latent_dim
        )
    
    elif name == 'conv':
        return ConvEncoder(latent_dim=latent_dim,
                           dataset = cfg['dataset']['name'],
                           use_norm=encoder_cfg['use_norm'],
                           bias = encoder_cfg['bias'])
   
    elif name == 'mlp':
          return MLPEncoder(
                input_dim=encoder_cfg['input_dim'][cfg['dataset']['name']],
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
    print("decoder_latent_dim", latent_dim)
    if name == 'linear':
        if cfg['dataset']['name'] in ['SVHN', 'CIFAR10', 'CelebA','CIFAR16', 'CelebA64', 'FFHQ']:
            return LinearDecoder(
                latent_dim=latent_dim,
                output_dim=cfg['encoder']['input_dim'][cfg['dataset']['name']],
                tanh=True
            )
        else:
            return LinearDecoder(
                latent_dim=latent_dim,
                output_dim=cfg['encoder']['input_dim'][cfg['dataset']['name']]
            )
    elif name == 'conv':
        if cfg['dataset']['name'] in ['SVHN', 'CIFAR10', 'CelebA','CIFAR16','CelebA64', 'FFHQ']:
             return ConvDecoder(latent_dim=latent_dim, 
                            out_channels=cfg['decoder']['out_channel'][cfg['dataset']['name']],
                            size=cfg['decoder']['size'][cfg['dataset']['name']],
                            tanh=True
            )
        else:
            return ConvDecoder(latent_dim=latent_dim, 
                            out_channels=cfg['decoder']['out_channel'][cfg['dataset']['name']],
                            size=cfg['decoder']['size'][cfg['dataset']['name']]
            )
    elif name == 'mlp':
        if cfg['dataset']['name'] in ['SVHN', 'CIFAR10', 'CelebA','CIFAR16','CelebA64', 'FFHQ']:
            return MLPDecoder(
                        latent_dim=latent_dim,
                        output_dim=cfg['encoder']['input_dim'][cfg['dataset']['name']],
                        normalize=decoder_cfg.get('normalize', False),
                        bias=decoder_cfg.get('bias', False),
                        activation_fn=decoder_cfg.get('activation_fn', 'swish'),
                        tanh=True
                )
        else:
            return MLPDecoder(
                    latent_dim=latent_dim,
                    output_dim=cfg['encoder']['input_dim'][cfg['dataset']['name']],
                    normalize=decoder_cfg.get('normalize', False),
                    bias=decoder_cfg.get('bias', False),
                    activation_fn=decoder_cfg.get('activation_fn', 'swish'),
            )
    else:
        raise ValueError(f"Unsupported decoder type: {name}")
    

def get_overdispersion_index(z, eps=1e-8):
    if isinstance(z, torch.Tensor):
        z_mean = z.mean(dim=0)
        z_var = z.var(dim=0, unbiased=False)
    else:
        z_mean = np.mean(z, axis=0)
        z_var = np.var(z, axis=0)

    return ((z_var + eps) / (z_mean + eps)).mean()



def tonp(x: Union[torch.Tensor, np.ndarray]):
	if isinstance(x, np.ndarray):
		return x
	elif isinstance(x, torch.Tensor):
		return x.data.cpu().numpy()
	else:
		raise ValueError(type(x).__name__)
     

def find_last_contiguous_zeros(mask: np.ndarray, w: int):
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


def softclamp_sym(x, clamp=5.3):
    return clamp * torch.tanh(x / clamp)



def to_cpu_float64(x):
    return x.to(dtype=torch.float64, device="cpu")

def _select_and_stack(tensor_list, max_samples, normalize=False):
    selected = []
    total = 0
    for t in tensor_list:
        if t is None:
            continue
        if total + t.size(0) > max_samples:
            selected.append(t[:max_samples - total])
            break
        selected.append(t)
        total += t.size(0)

    if not selected:
        return None

    out = torch.cat(selected, dim=0)
    if normalize:
        out = ((out + 1) / 2).clamp(0, 1)
    else:
        out = out.clamp(0, 1)
    return out
