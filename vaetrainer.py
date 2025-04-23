import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.loggers import wandb
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets

import wandb as wdb
from model import GenericVAE
from utils import (build_decoder, build_encoder, find_last_contiguous_zeros,
                   get_overdispersion_index, log_dead_neurons_diagnostics,
                   log_latent_mean_vs_var, plot_fano, spike_count_hist)


class VAETrainer(pl.LightningModule):
    def __init__(self, cfg, **args):
        super(VAETrainer, self).__init__()
        self.cfg = cfg 
        encoder = build_encoder(cfg)
        decoder = build_decoder(cfg)
       
        self.model = GenericVAE(
                encoder=encoder,
                decoder=decoder,
                latent_dim=cfg['model']['latent_dim'],
                dist_type=cfg['model']['name'],
                reparam_type = cfg['model']['reparam_type'],
                max_count = cfg['model']['max_count'],
                tau = cfg['model']['tau'],
            )
        self.model_name = cfg['model']['name']

        opt_name = cfg['optimizer']['name'].lower()
        opt_map = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        self.opt = opt_map[opt_name]
        self.opt_params = {k: v for k, v in cfg['optimizer'].items() if k != 'name'}

        self.beta = 0.0
        self.train_length = None

        fid_feat_dim = cfg.get('eval', {}).get('fid_feature', 64)
        self.fid_metric = FrechetInceptionDistance(feature=fid_feat_dim, reset_real_features=True, normalize=True)
        self.log_images = cfg.get('eval', {}).get('log_images', True)
        self._val_kl_diags = []
        self._val_latents = []

    def setup(self, stage = None):
        if hasattr(self.trainer.datamodule, 'train_steps_per_epoch'):
            self.train_length = self.trainer.datamodule.train_steps_per_epoch

            # for batch in self.trainer.datamodule.val_dataloader():
            #     real_imgs = batch[0].repeat(1, 3, 1, 1)
            #     self.fid_metric.update(real_imgs, real=True)
        else:
            raise ValueError("DataModule not defined train_steps_per_epoch")


    def forward(self, x):
        if self.cfg['encoder']['type']=="linear":
            return self.model(x[0].flatten(1))
        else:
            return self.model(x[0])
    
    def training_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1) 
        epoch = self.current_epoch + batch_idx/self.train_length
        self.beta = min(1.0, 5*epoch/250)
        self.model.t = max((1.0 - 0.95*epoch/250), 0.05)
        self.log('beta', self.beta)
        self.log('t', self.model.t)

        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            kl = dist.kl(self.model.prior, du).mean()
        elif self.model_name == "negbio":
            dist, logit_p, z, y = self(batch)    
            if self.cfg['model']['kl'] == "mc":
                # print("MC estimation")
                kl =  self.model.dist_class.kl_mc(self.model.log_r_prior, 
                                          self.model.logit_p_prior, 
                                          logit_p
                                        ).mean()
            else:
                kl = self.model.dist_class.kl(self.model.log_r_prior, 
                                          self.model.logit_p_prior, 
                                          logit_p
                                        ).mean()

      
        if self.cfg['decoder']['type']=="conv":
            x = batch[0].view(-1, 1, 28, 28)

    
        recon_loss = self.model.mse_loss(x, y)

        loss = self.beta*kl + recon_loss 

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_elbo', (kl + recon_loss).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("latent mean", (z.mean()).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("latent std", (z.std()).item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1) 

        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            kl_diag = dist.kl(self.model.prior, du) 
            # kl = dist.kl(self.model.prior, du).mean()
        elif self.model_name == "negbio":
            dist, logit_p, z, y = self(batch)
            kl_diag = self.model.dist_class.kl(
                self.model.log_r_prior, 
                self.model.logit_p_prior, 
                logit_p
                )
        kl = kl_diag.mean()

        if self.cfg['decoder']['type']=="conv":
            x = x.view(-1, 1, 28, 28)

        recon_loss = self.model.mse_loss(x, y)

     
        val_elbo = recon_loss + kl

        loss = self.beta*kl + recon_loss
        overdispersion_index = get_overdispersion_index(z)

        self.log('val_recon_loss', recon_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl', kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo', val_elbo.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo_mc', (kl + recon_loss).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('l0_sparsity', (z == 0).float().mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('overdispersion_index', overdispersion_index, on_step=True, on_epoch=True, prog_bar=True)

        if self.cfg['datasetname'] == 'MNIST':
            real_imgs = batch[0].repeat(1, 3, 1, 1)  # MNIST 是 1 通道，要扩成 3 通道
            recon_imgs = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
        elif self.cfg['datasetname'] == 'CIFAR16':
            real_imgs = batch[0].view(-1, 3, 16, 16)
            recon_imgs = y.view(-1, 3, 16, 16)

        self._val_kl_diags.append(kl_diag.detach().cpu()) 
        self._val_latents.append(z.detach().cpu())
        self.fid_metric.update(real_imgs, real=True)
        self.fid_metric.update(recon_imgs, real=False)

        if batch_idx % 50 == 0:
            if self.cfg['datasetname'] == 'MNIST':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 1, 28, 28), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 1, 28, 28), nrow=10)
            elif self.cfg['datasetname'] == 'CIFAR16':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 16, 16), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 16, 16), nrow=10)
            else:
                raise ValueError(f"Unseen dataset name: {self.cfg['dataset']['name']}")
            self.logger.experiment.log({
                'recons': wdb.Image(fig_y, caption="recons"),
                'inputs': wdb.Image(fig_x, caption="inputs"),
            })
            log_latent_mean_vs_var(
                logger=self.logger.experiment, 
                z=z, 
                save_dir=None,
                step_name=f"val_epoch_{self.current_epoch}", 
                caption="Latent mean vs var",
                savelocal=False
            )
        
        return loss
    
    def on_validation_epoch_end(self):

        if self.current_epoch % 20 ==0 or self.current_epoch == self.trainer.max_epochs - 1:
            if len(self._val_kl_diags) > 0:
                kl_diag = torch.cat(self._val_kl_diags, dim=0).mean(dim=0).numpy()  # shape: [latent_dim]
                dead_mask = self.find_dead_neurons(kl=kl_diag)

                self.log("num_dead_units", dead_mask.sum().item(), prog_bar=True)
                self.logger.experiment.log({"num_dead_units": dead_mask.sum().item()})
                log_dead_neurons_diagnostics(
                    kl_diag=kl_diag,
                    dead_mask=dead_mask,
                    logger=self.logger,
                    step_name=f"val_epoch_{self.current_epoch}"
                )
            else:
                print("[Warning] No KL diagnostics found for this epoch.")
            
            #self._val_latents = []
            # self._val_kl_diags = []   # empty the kl_diags
            
            fid_score = self.fid_metric.compute().item()
            self.log("val_fid", fid_score, prog_bar=True)
            self.logger.experiment.log({"val_fid": fid_score})
            self.fid_metric.reset()
            
    def configure_optimizers(self):
        return self.opt(self.parameters(), **self.opt_params)
    

    def find_dead_neurons(self, kl: np.ndarray = None):
        eps = np.finfo(kl.dtype).eps
        kl = np.maximum(kl, eps)

        cfg = self.cfg['model']
        enc_type = self.cfg['encoder']['type']
        dataset = self.cfg['dataset']['name'] if 'data' in self.cfg else self.cfg.get('dataset', 'Unknown')
        model_type = cfg['name']

        rules = {
        ('pm', 1e-2): (model_type == 'poisson' and dataset == 'MNIST'),
        ('nb', 1e-2): (model_type == 'negbio' and dataset == 'MNIST'),
        ('ll', 1e-1): (model_type == 'laplace' and enc_type == 'linear'),
        ('gl', 1e-1): (model_type == 'gaussian' and enc_type == 'linear' and dataset != 'CIFAR10-PATCHES'),
        ('glc', 85e-3): (model_type == 'gaussian' and enc_type == 'linear' and dataset == 'CIFAR10-PATCHES'),
        }
        for (_, thres), matched in rules.items():
            if matched:
                return kl < thres

            
        if model_type == 'categorical' and hasattr(self.model, 'find_dead_neurons'):
            return self.model.find_dead_neurons(2)
        
        if enc_type == 'linear' and hasattr(self.model, 'find_dead_neurons'):
            dead = self.model.find_dead_neurons(8)
            order = np.argsort(kl)
            idx = np.argmax(~dead[order])
            dead = np.zeros(len(dead))
            dead[order[:idx]] = 1
            return dead.astype(bool)
        
        log_kl = np.log(kl)
        bins = np.linspace(start=np.nanmin(log_kl), stop=np.nanmax(log_kl), num=len(log_kl) * 10)
        hist, _ = np.histogram(log_kl, bins=bins)

        idx = find_last_contiguous_zeros(mask=hist > 0, w=len(log_kl) * 2)
        dead = log_kl < bins[idx] if idx is not None else kl < 3e-4
        return dead.astype(bool)


        
    # def log_dead_neurons_diagnostics(self, kl_diag, dead_mask, step_name="val"):
    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     dims = np.arange(len(kl_diag))
    #     kl_vals = kl_diag
    #     ax.bar(dims[dead_mask], kl_vals[dead_mask], color="red", label="Dead neuron")
    #     ax.bar(dims, kl_vals, color="blue", label="KL per dim")
    #     ax.set_xlabel("Latent Dimension")
    #     ax.set_ylabel("KL Divergence")
    #     ax.set_title(f"[{step_name}] KL per latent dim")
    #     ax.legend()
    #     plt.tight_layout()

    #     self.logger.experiment.log({
    #         f"kl_per_dim": wdb.Image(fig, caption="KL per latent dim (red = dead)"),
    #     })
    #     plt.close(fig)

    def on_fit_end(self, end=False):

        # Build save dirs
        if end:
            save_dirs = {
                "latents": os.path.join(self.logger.save_dir, "latents", self.logger.experiment.name),
                "fano": os.path.join(self.logger.save_dir, "analysis", "fano", self.logger.experiment.name),
                "meanvar": os.path.join(self.logger.save_dir, "analysis", "meanvar", self.logger.experiment.name),
                "spike_hist": os.path.join(self.logger.save_dir, "analysis", "spike_hist", self.logger.experiment.name),
            }
        else:
            save_dirs = {
                "latents": os.path.join(".save/", "latents"),
                "fano": os.path.join(".save/", "analysis", "fano"),
                "meanvar": os.path.join(".save/", "analysis", "meanvar"),
                "spike_hist": os.path.join(".save/", "analysis", "spike_hist"),
            }

        for path in save_dirs.values():
            os.makedirs(path, exist_ok=True)

        if len(self._val_latents) > 0:
            all_z = torch.cat(self._val_latents, dim=0).numpy()
            np.save(os.path.join(save_dirs["latents"], "val_z_all.npy"), all_z)
            print(f"[✔] Saved all validation z")
        else:
            print("[⚠] No validation z collected to save.")


        # ===== Overdispersion index & mean-var scatter =====
        overdispersion_index = get_overdispersion_index(all_z)
        self.logger.experiment.log({
            "final_overdispersion_index": overdispersion_index
        })

        log_latent_mean_vs_var(
            logger=self.logger.experiment,
            z=all_z,
            save_dir=save_dirs['meanvar'],
            step_name="final",
            caption="Latent Mean vs Variance"
        )

        #===== Fano plot=================
        plot_fano(all_z, save_dirs["fano"],self.logger)

        #===== Hist plot==================
        spike_count_hist(all_z, save_dirs['spike_hist'], self.logger, top_k=8)

        # ===== Dead units and KL diagnostic =====

        kl_diag = torch.cat(self._val_kl_diags, dim=0).mean(dim=0).numpy()
        dead_mask = self.find_dead_neurons(kl=kl_diag)

        log_dead_neurons_diagnostics(
                kl_diag=kl_diag,
                dead_mask=dead_mask,
                logger=self.logger,
                step_name=f"val_epoch_{self.current_epoch}"
            )







        




