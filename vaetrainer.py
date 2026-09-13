import os
import warnings
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data import DataModule
from clf_analysis import train_clf_analysis
warnings.filterwarnings("ignore")
import wandb as wdb
from model import GenericVAE
from utils import (_select_and_stack, build_decoder, build_encoder,
                   compute_inception_score, find_last_contiguous_zeros,
                   get_overdispersion_index)


class VAETrainer(pl.LightningModule):
    def __init__(self, cfg, **args):
        super(VAETrainer, self).__init__()
        self.cfg = cfg
        encoder = build_encoder(cfg)
        decoder = build_decoder(cfg)

        self.model_name = cfg['model']['name']
        base_latent_dim = cfg['model']['latent_dim']


        self.model = GenericVAE(
                encoder=encoder,
                decoder=decoder,
                latent_dim=base_latent_dim,
                dist_type=cfg['model']['name'],
                reparam_type = cfg['model']['reparam_type'],
                max_count = cfg['model']['max_count'],
                tau = cfg['model']['tau'],
                latent_act= cfg['model']['latent_act'],
                num_samples = cfg['model']['num_samples'],
                kl_type = cfg['model']['kl'],
                cts_max_count = cfg['model'].get('cts_max_count', 64),
            )

        self.log_images = self.cfg.get('eval', {}).get('log_images', True)
        opt_name = cfg['optimizer']['name'].lower()
        opt_map = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        self.opt = opt_map[opt_name]
        self.opt_params = {k: v for k, v in cfg['optimizer'].items() if k != 'name'}

        self.beta = self.cfg['model']['beta']
        self.kl_annealing = self.cfg['model']['kl_annealing']
        self.cch_detach_phi = self.cfg['model'].get('cch_detach_phi', False)
        self.train_length = None


        self._val_kl_diags = []
        self._val_latents = []
        self._val_recons = []
        self._val_inputs = []
        self._val_oi_list = []
        self._val_dnr = None
        self._final_val_fid = None

        self._test_latents = []
        self._test_labels = []
        self._test_recons = []
        self._test_inputs = []
        self._test_recon_terms = [] 
        self._test_kl_terms = [] 




    def setup(self, stage = None):
        if hasattr(self.trainer.datamodule, 'train_steps_per_epoch'):
            self.train_length = self.trainer.datamodule.train_steps_per_epoch
        else:
            raise ValueError("DataModule not defined train_steps_per_epoch")


    def forward(self, x):
        if self.cfg['encoder']['type']=="linear":
            return self.model(x[0].flatten(1))
        else:
            return self.model(x[0])

    def _negative_binomial_kl(self, log_r_post, logit_p_post, z,
                              return_components=False):
        kl_type = self.cfg['model']['kl']
        if kl_type == "mc":
            kl = self.model.dist_class.kl_mc(
                self.model.log_r_prior,
                self.model.logit_p_prior,
                log_r_post,
                logit_p_post,
            )
            return (kl, None) if return_components else kl
        if kl_type == "cch":
            return self.model.dist_class.kl_cch(
                self.model.log_r_prior,
                self.model.logit_p_prior,
                log_r_post,
                logit_p_post,
                z,
                detach_phi=self.cch_detach_phi,
                return_components=return_components,
            )
        kl = self.model.dist_class.kl(
            self.model.log_r_prior,
            self.model.logit_p_prior,
            logit_p_post,
        )
        return (kl, None) if return_components else kl

    def _reduce_kl(self, kl_diag):
        if self.cfg['model']['kl'] == "cch":
            return kl_diag.flatten(1).sum(dim=1).mean()
        return kl_diag.mean()

    def training_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1)

        epoch = self.current_epoch + batch_idx/self.train_length
        if self.kl_annealing:
            self.beta =  min(1.0, 5*epoch/250) 
        if self.cfg['model']['kl'] == "cch":
            # CCH accuracy relies on a low CTS temperature. Do not overwrite
            # the configured value with the legacy 1.0 -> 0.05 schedule.
            self.model.t = self.model.tau
        else:
            self.model.t = max((1.0 - 0.95*epoch/250), 0.05)
        self.log('beta', self.beta)
        self.log('t', self.model.t)

        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            kl = dist.kl(self.model.prior, du).mean()
        elif self.model_name == "negbio":
            dist, (log_r_post, logit_p), z, y = self(batch)
            kl_diag, cch_components = self._negative_binomial_kl(
                log_r_post, logit_p, z, return_components=True
            )
            kl = self._reduce_kl(kl_diag)
            if cch_components is not None:
                kl_gamma = cch_components['kl_gamma'].mean()
                phi = cch_components['phi'].mean()
                phi_ratio = phi / kl_gamma.clamp_min(1e-8)
                clamp_rate = (cch_components['unclamped'] < 0).float().mean()
                truncation_rate = self.model.dist_class.strategy.last_truncation_rate
                self.log('cch_kl_gamma', kl_gamma, on_step=True, on_epoch=True)
                self.log('cch_phi', phi, on_step=True, on_epoch=True)
                self.log('cch_phi_ratio', phi_ratio, on_step=True, on_epoch=True)
                self.log('cch_clamp_rate', clamp_rate, on_step=True, on_epoch=True)
                self.log('cch_cts_truncation_rate', truncation_rate,
                         on_step=True, on_epoch=True)

        elif self.model_name == "categorical":
            dist, logit_p, z, y = self(batch)
            kl = self.model.dist_class.comput_kl(logit_p).mean()

        elif self.model_name == "laplace":
            dist, (loc, log_scale), z, y = self(batch)
            kl = dist.kl().mean()

        elif self.model_name == "gaussian":
            dist, (loc, log_scale), z, y = self(batch)
            kl = dist.kl().mean()

        if self.cfg['decoder']['type']=="conv":
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                x = batch[0].view(-1, 1, 28, 28)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                x = batch[0].view(-1, 3, 16, 16)
            elif self.cfg['dataset']['name'] == 'SVHN':
                x = batch[0].view(-1, 3, 32, 32)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']:
                x = batch[0].view(-1, 3, 64, 64)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                x = batch[0].view(-1, 3, 128, 128)



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
        elif self.model_name == "negbio":
            dist, (log_r_post, logit_p), z, y = self(batch)
            kl_diag = self._negative_binomial_kl(log_r_post, logit_p, z)
        elif self.model_name == "categorical":
            dist, logit_p, z, y = self(batch)
            kl_diag = self.model.dist_class.comput_kl(logit_p)
        elif self.model_name == "laplace":
            dist, (loc, log_scale), z, y = self(batch)
            kl_diag = dist.kl()
        elif self.model_name == "gaussian":
            dist, (loc, log_scale), z, y = self(batch)
            kl_diag = dist.kl()

        kl = self._reduce_kl(kl_diag)
        if self.cfg['decoder']['type']=="conv":
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                x = x.view(-1, 1, 28, 28)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                x = x.view(-1, 3, 16, 16)
            elif self.cfg['dataset']['name'] == 'SVHN':
                x = x.view(-1, 3, 32, 32)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']:
                x = x.view(-1, 3, 64, 64)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                x = x.view(-1, 3, 128, 128)

        num_dims = np.prod(x.shape[1:])
        recon_loss = self.model.mse_loss(x, y)
        val_elbo = recon_loss + kl
        loss = self.beta*kl + recon_loss  
        overdispersion_index = get_overdispersion_index(z)
        self._val_oi_list.append(overdispersion_index)
        self.log('val_recon_loss', recon_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl', kl.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo', val_elbo.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_elbo_mc', (kl + recon_loss).item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('l0_sparsity', (z == 0).float().mean().item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('overdispersion_index', overdispersion_index, on_step=True, on_epoch=True, prog_bar=True)
   

        if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
            real_imgs = batch[0].repeat(1, 3, 1, 1)  
            recon_imgs = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
        elif self.cfg['dataset']['name'] == 'CIFAR16':
            real_imgs = batch[0].view(-1, 3, 16, 16)
            recon_imgs = y.view(-1, 3, 16, 16)
        elif self.cfg['dataset']['name'] == 'SVHN':
            real_imgs = batch[0].view(-1, 3, 32, 32)
            recon_imgs = y.view(-1, 3, 32, 32)
        elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']:
            real_imgs = batch[0].view(-1, 3, 64, 64)
            recon_imgs = y.view(-1, 3, 64, 64)
        elif self.cfg['dataset']['name'] in ['CelebAHQ']:
            real_imgs = batch[0].view(-1, 3, 128, 128)
            recon_imgs = y.view(-1, 3, 128, 128)

        
        self._val_kl_diags.append(kl_diag.detach().cpu())
        self._val_latents.append(z.detach().cpu())
        self._val_recons.append(recon_imgs.detach().cpu())
        self._val_inputs.append(real_imgs.detach().cpu())

        if batch_idx % 100 == 0:
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 1, 28, 28), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 1, 28, 28), nrow=10)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 16, 16), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 16, 16), nrow=10)
            elif self.cfg['dataset']['name'] == 'SVHN':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 32, 32), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 32, 32), nrow=10)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']:  # 
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 64, 64), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 64, 64), nrow=10)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 128, 128),nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 128, 128), nrow=10)


            else:
                raise ValueError(f"Unseen dataset name: {self.cfg['dataset']['name']}")
            if self.logger is not None:
                self.logger.experiment.log({
                    'recons': wdb.Image(fig_y, caption="recons"),
                    'inputs': wdb.Image(fig_x, caption="inputs"),
                })

        return loss

    def on_validation_epoch_end(self):

        if self.current_epoch != self.trainer.max_epochs - 1:
            return

        if len(self._val_kl_diags) > 0:
            kl_diag = torch.cat(self._val_kl_diags, dim=0).mean(dim=0).numpy()
            dead_mask = self.find_dead_neurons(kl=kl_diag)
            dnr = dead_mask.mean().item()
            self._val_dnr = dnr

            self.log("num_dead_units", dead_mask.sum().item(), prog_bar=True)
            if self.logger is not None:
                self.logger.experiment.log({"num_dead_units": dead_mask.sum().item()})
        else:
            print("[Warning] No KL diagnostics found for this epoch.")

        self.log("dead_neuron_rate",dnr, prog_bar=True)
        if self.logger is not None:
            self.logger.experiment.log({"dnr": dnr})

        if len(self._val_oi_list) > 0:
            avg_oi = sum(self._val_oi_list) / len(self._val_oi_list)
            self.log("overdispersion_index_epoch", avg_oi)
            if self.logger is not None:
                self.logger.experiment.log({"epoch_oi": avg_oi})
            self.oi = avg_oi
        else:
            print("[⚠] No OI values collected.")


        #=========test=============================
        if len(self._val_latents) > 0:
            last_20 = self._val_latents[-5:]
            self.save_val_z = torch.cat(last_20, dim=0)
        else:
            self.save_val_z = torch.cat(self._val_latents, dim=0)  

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
        ('pm', 1e-2): (model_type == 'poisson' and dataset in ['MNIST', "Omniglot","fmnist"]),
        ('nb', 1e-2): (model_type == 'negbio' and dataset in ['MNIST', "Omniglot","fmnist"]),
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


    def build_fid_reference(self,dataset_name,fid_metric,device,batch_size):
        if self.cfg['encoder']['type'] == "conv":
            flatten_flag = False
        else:
            flatten_flag = True
        if dataset_name.lower() in ["cifar10","cifar16"]:
            dm = DataModule(dataset_name=dataset_name,batch_size=batch_size,flatten=flatten_flag)
            dm.prepare_data()
            dm.setup()
            ref_dataset = torch.utils.data.ConcatDataset([dm.train_set, dm.val_set])
        elif dataset_name.lower() in ["celeba","celeba64"]:
            dm = DataModule(dataset_name=dataset_name,batch_size=batch_size,flatten=flatten_flag)
            dm.prepare_data()
            dm.setup()
            ref_dataset = torch.utils.data.ConcatDataset([dm.train_set, dm.val_set, dm.test_set])
        else:
            raise ValueError(f"unsupported dataset")
        
        loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch, _ in loader:
                fid_metric.update(batch.to(device), real=True)

        print(f"[✔] Built FID reference for {dataset_name} ({len(ref_dataset)} images).")
        return fid_metric



    def test_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1)
        gt_label = batch[1]
        if self.model_name == "poisson":
            dist, du, z, y = self(batch)
            z_repr = du
            kl_diag = dist.kl(self.model.prior, du)
        elif self.model_name == "negbio":
            dist, (log_r_post, logit_p), z, y = self(batch)
            if self.cfg['model']['kl'] == "cch":
                # Posterior Gamma/CTS mean alpha_q / beta_q is a D-vector.
                z_repr = torch.exp((log_r_post - logit_p).clamp(-10, 10))
            else:
                z_repr = logit_p
            kl_diag = self._negative_binomial_kl(log_r_post, logit_p, z)
        elif self.model_name == "categorical":
            dist, logit_p, z, y = self(batch)
            z_repr = logit_p
            kl_diag = self.model.dist_class.comput_kl(logit_p)
        elif self.model_name == "laplace":
            dist, (loc, log_scale), z, y = self(batch)
            z_repr = loc
            kl_diag = dist.kl()
        elif self.model_name == "gaussian":
            dist, (loc, log_scale), z, y = self(batch)
            z_repr = loc
            kl_diag = dist.kl()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.cfg['decoder']['type']=="conv":
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                x = x.view(-1, 1, 28, 28)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                x = x.view(-1, 3, 16, 16)
            elif self.cfg['dataset']['name'] == 'SVHN':
                x = x.view(-1, 3, 32, 32)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']: 
                x = x.view(-1, 3, 64, 64)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                x = x.view(-1, 3, 128, 128)

        recon_loss = self.model.mse_loss(x, y)
        per_sample_recon = F.mse_loss(x, y, reduction='none')
        per_sample_recon = per_sample_recon.view(per_sample_recon.size(0), -1).sum(dim=1)
        if kl_diag.dim() > 1:
            per_sample_kl = kl_diag.sum(dim=1)
        else:
            per_sample_kl = kl_diag
        per_sample_nll = per_sample_recon + per_sample_kl

        self._test_recon_terms.append(per_sample_recon.detach().cpu())
        self._test_kl_terms.append(per_sample_kl.detach().cpu())
        self._test_latents.append(z_repr.detach().cpu())
        self._test_labels.append(gt_label.detach().cpu())


        if self.cfg['encoder']['type'] == "conv":
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                real_imgs = batch[0].repeat(1, 3, 1, 1)
                recon_imgs = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                real_imgs = batch[0].view(-1, 3, 16, 16)
                recon_imgs = y.view(-1, 3, 16, 16)
            elif self.cfg['dataset']['name'] == 'SVHN':
                real_imgs = batch[0].view(-1, 3, 32, 32)
                recon_imgs = y.view(-1, 3, 32, 32)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']: 
                real_imgs = batch[0].view(-1, 3, 64, 64)
                recon_imgs = y.view(-1, 3, 64, 64)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                real_imgs = batch[0].view(-1, 3, 128, 128)
                recon_imgs = y.view(-1, 3, 128, 128)


        else:
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                real_imgs = batch[0].view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
                recon_imgs = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                real_imgs = batch[0].view(-1, 3, 16, 16)
                recon_imgs = y.view(-1, 3, 16, 16)
            elif self.cfg['dataset']['name'] == 'SVHN':
                real_imgs = batch[0].view(-1, 3, 32, 32)
                recon_imgs = y.view(-1, 3, 32, 32)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']: 
                real_imgs = batch[0].view(-1, 3, 64, 64)
                recon_imgs = y.view(-1,3,64,64)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                real_imgs = batch[0].view(-1, 3, 128, 128)
                recon_imgs = y.view(-1, 3, 128, 128)

            else:
                raise ValueError(f"Unsupported dataset: {self.cfg['dataset']['name']}")


        self._test_recons.append(recon_imgs.detach().cpu())
        self._test_inputs.append(real_imgs.detach().cpu())

        if batch_idx == 0:
            if self.cfg['dataset']['name'] in ['MNIST', "Omniglot","fmnist"]:
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 1, 28, 28), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 1, 28, 28), nrow=10)
            elif self.cfg['dataset']['name'] == 'CIFAR16':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 16, 16), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 16, 16), nrow=10)
            elif self.cfg['dataset']['name'] == 'SVHN':
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 32, 32), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 32, 32), nrow=10)
            elif self.cfg['dataset']['name'] in ['CelebA', 'CelebA64', 'FFHQ']: 
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 64, 64), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 64, 64), nrow=10)
            elif self.cfg['dataset']['name'] in ['CelebAHQ']:
                fig_y = torchvision.utils.make_grid(y.reshape(-1, 3, 128, 128), nrow=10)
                fig_x = torchvision.utils.make_grid(batch[0].reshape(-1, 3, 128, 128), nrow=10)


            else:
                raise ValueError(f"Unseen dataset name: {self.cfg['dataset']['name']}")
            if self.logger is not None:
                self.logger.experiment.log({
                    'test_recons': wdb.Image(fig_y, caption="test_recons"),
                    'test_inputs': wdb.Image(fig_x, caption="test_inputs"),
                })

        return recon_loss.detach()


    def on_test_end(self, end=False):
        if end:
            save_root = self.logger.save_dir
            if self.logger is not None:
                experiment_name = self.logger.experiment.name
            save_dirs = {
                "latents": os.path.join(save_root, "latents", experiment_name),
                "fano": os.path.join(save_root, "analysis", "fano", experiment_name),
                "meanvar": os.path.join(save_root, "analysis", "meanvar", experiment_name),
                "spike_hist": os.path.join(save_root, "analysis", "spike_hist", experiment_name),
            }

        else:
            prefix = ["save", self.cfg['dataset']['name'], self.cfg['model']['name'],
                    self.cfg['model']['kl'], self.cfg['model']['reparam_type']]
            save_dirs = {
                "latents": os.path.join(*prefix, "latents"),
                "fano": os.path.join(*prefix, "analysis", "fano"),
                "meanvar": os.path.join(*prefix, "analysis", "meanvar"),
                "spike_hist": os.path.join(*prefix, "analysis", "spike_hist"),
            }

        for path in save_dirs.values():
            os.makedirs(path, exist_ok=True)

        if len(self._test_latents) > 0:
            all_z = torch.cat(self._test_latents, dim=0).numpy()
            all_gt_label = torch.cat(self._test_labels, dim=0).numpy()
            if self.cfg['logging']['save_files']:
                np.save(os.path.join(save_dirs["latents"], 
                                    "test_z_all_{}.npy".format(self.cfg['model']['latent_dim'])), 
                                    all_z)
                np.save(os.path.join(save_dirs['latents'], 
                                    "test_y_all_{}.npy".format(self.cfg['model']['latent_dim'])), 
                                    all_gt_label)
                print(f"[✔] Saved all test rep")
                print(f"[✔] Saved all test label")
        else:
            print("[⚠] No test z collected to save.")
            return
     
        if self.cfg['logging']['save_files']:
            np.save(os.path.join(save_dirs["latents"], 
                                    "val_spike_all_{}.npy".format(self.cfg['model']['latent_dim'])), 
                                    self.save_val_z)
            print(f"[✔] Saved all val spikes")


        max_samples = 5000 
        normalize_needed = self.cfg['dataset']['name'].lower() in ['svhn', 'cifar10', 'celeba','cifar16','celeba64','celebahq']

        recon_sample = _select_and_stack(self._test_recons, max_samples, normalize=normalize_needed)
        input_sample = _select_and_stack(self._test_inputs, max_samples, normalize=normalize_needed)

        if recon_sample is not None:
            recon_sample = recon_sample.to(self.device)
        if input_sample is not None:
            input_sample = input_sample.to(self.device)

        # ========== MSE ==========
        if input_sample is not None and recon_sample is not None:
            mse_mean = F.mse_loss(recon_sample, input_sample).item()
        else:
            mse = None
            print("[⚠] MSE skipped due to missing recon/input.")

        # ========== PRINT Final Results ==========
        print("\n=== 🎯 Final Performance | Dataset: {} | Model: {} ===".format(self.cfg['dataset']['name'], self.cfg['model']['name']))
        if mse_mean is not None:
            print(f"[📐] MSE_MEAN: {mse_mean:>23.4f}") 
        # ========== WandB Logging ==========
        log_dict = {
            "final_dnr": self._val_dnr
        }
        if mse_mean is not None:
            log_dict["final_mse"] = mse_mean
        if self.logger is not None:
            self.logger.experiment.log(log_dict)
