import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision, math, os, torch
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from model import GenericVAE
from utils import build_decoder, build_encoder




class VAETrainer(pl.LightningModule):
    def __init__(self, cfg, **args):
        super(VAETrainer, self).__init__()
        self.cfg = cfg 
        encoder = build_encoder(cfg['encoder'])
        decoder = build_decoder(cfg['decoder'])
       
        self.model = GenericVAE(
                encoder=encoder,
                decoder=decoder,
                latent_dim=cfg['model']['latent_dim'],
                dist_type=cfg['model']['name']
            )
        self.model_name = cfg['model']['name']

        opt_name = cfg['optimizer']['name'].lower()
        opt_map = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        self.opt = opt_map[opt_name]
        self.opt_params = {k: v for k, v in cfg['optimizer'].items() if k != 'name'}

        self.beta = 0.0
        self.train_length = None

        fid_feat_dim = cfg.get('eval', {}).get('fid_feature', 64)
        self.fid_metric = FrechetInceptionDistance(feature=fid_feat_dim, normalize=True)
        self.log_images = cfg.get('eval', {}).get('log_images', True)

    def setup(self, stage = None):
        if hasattr(self.trainer.datamodule, 'train_steps_per_epoch'):
            self.train_length = self.trainer.datamodule.train_steps_per_epoch
        else:
            raise ValueError("DataModule not defined train_steps_per_epoch")


    def forward(self, x):
        return self.model(x[0].flatten(1))
    
    def training_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1) 
        epoch = self.current_epoch + batch_idx/self.train_length
        self.beta = min(5.0, 5*epoch/250)
        self.model.t = max((1.0 - 0.95*epoch/250), 0.05)
        self.log('beta', self.beta)
        self.log('t', self.model.t)

        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            kl = dist.kl(self.model.prior, du).mean()
        elif self.model_name == "negbio":
            dist, logit_p, z, y = self(batch)
            kl = dist.kl(self.model.log_r_prior, self.model.logit_p_prior).mean()

        mse = ((y - x)**2).sum(-1).mean()
        loss = self.beta*kl + mse
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_elbo', (kl + mse).item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1) 

        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            kl = dist.kl(self.model.prior, du).mean()
        elif self.model_name == "negbio":
            dist, logit_p, z, y = self(batch)
            kl = dist.kl(self.model.log_r_prior, self.model.logit_p_prior).mean()

        mse = ((y - x)**2).sum(-1).mean()
        loss = self.beta*kl + mse

        self.log('val_mse', mse.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl', kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo', (kl + mse).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('l0_sparsity', (z == 0).float().mean().item(), on_step=True, on_epoch=True, prog_bar=True)

        real_imgs = batch[0].repeat(1, 3, 1, 1)  # MNIST 是 1 通道，要扩成 3 通道
        recon_imgs = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)

        self.fid_metric.update(real_imgs, real=True)
        self.fid_metric.update(recon_imgs, real=False)

        if batch_idx == 0:
            fig_y = torchvision.utils.make_grid(y.reshape(-1, 1, 28, 28), nrow=10)
            fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 1, 28, 28), nrow=10)
            self.logger.experiment.log({
                'recons': wdb.Image(fig_y, caption="recons"),
                'inputs': wdb.Image(fig_x, caption="inputs"),
            })
        return loss
    
    def on_validation_epoch_end(self):
        fid_score = self.fid_metric.compute().item()
        self.log("val_fid", fid_score, prog_bar=True)
        self.logger.experiment.log({"val_fid": fid_score})
        self.fid_metric.reset()
        
    def configure_optimizers(self):
        return self.opt(self.parameters(), **self.opt_params)
    
    def on_validation_epoch_end(self):
        fid_score = self.fid_metric.compute().item()
        self.log("val_fid", fid_score, prog_bar=True)
        self.logger.experiment.log({"val_fid": fid_score})
        self.fid_metric.reset()


