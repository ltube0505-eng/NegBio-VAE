import matplotlib.pyplot as plt

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
    z_mean = z.mean(dim=0)
    z_var = z.var(dim=0)
    eps = 1e-8

    # Plot mean vs var
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(z_mean.cpu(), z_var.cpu(), alpha=0.6, label='Latent units')
    ax.plot([0, z_mean.cpu().max()], [0, z_mean.cpu().max()], 'r--', label='Poisson (mean=var)')
    ax.set_xlabel('Mean of $z_i$')
    ax.set_ylabel('Variance of $z_i$')
    ax.set_title(f'[{step_name}] Latent Mean vs Variance')
    ax.legend()
    plt.tight_layout()

    # Compute average overdispersion index
    overdispersion_index = ((z_var + eps) / (z_mean + eps)).mean().item()

    logger.log({
        f"{step_name}_mean_vs_var": wdb.Image(fig, caption=caption),
        f"{step_name}_overdispersion_index": overdispersion_index
    })
    plt.close(fig)


def gumbel_entropy(y):
    # y: [B, D, K] softmax output
    return -(y * y.clamp(min=1e-8).log()).sum(dim=-1).mean()