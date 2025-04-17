import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

class GumbelMonitorCallback(pl.Callback):
    def __init__(self, log_every_n_steps=1):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.log_every_n_steps != 0:
            return

        model = pl_module.model
        if not hasattr(model, "gumbel_module"):
            return

        logit_p = model.logit_p  # Assuming exposed
        gumbel_module = model.gumbel_module

        with torch.no_grad():
            logits = gumbel_module.logits_layer(logit_p)
            logits = logits.view(-1, gumbel_module.latent_dim, gumbel_module.max_count)
            probs = torch.softmax(logits, dim=-1)
            entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=0)  # [latent_dim]
            z = gumbel_module(logit_p, hard=False)

        # Heatmap of entropy
        fig, ax = plt.subplots(figsize=(8, 2))
        sns.heatmap(entropy.unsqueeze(0).cpu().numpy(), ax=ax, cmap="viridis", cbar=True)
        ax.set_title("Gumbel Entropy per Latent Dim")
        ax.set_xlabel("Latent Dimensions")
        ax.set_ylabel("")
        trainer.logger.experiment.log({"gumbel_entropy": wandb.Image(fig)})
        plt.close(fig)

        # Z distribution summary
        trainer.logger.experiment.log({
            "z_mean": z.mean().item(),
            "z_std": z.std().item(),
            "tau": gumbel_module.tau
        })
