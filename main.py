import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from absl import app, flags
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, wandb

import wandb as wdb
from data import DataModule
from vaetrainer import VAETrainer

warnings.filterwarnings("ignore")
GPU = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] =GPU

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "configs/train_config.yaml", "Path to training config YAML file")
flags.DEFINE_string("reparam_type", "gumbel", "Reparameterization: gamma or gumbel")
flags.DEFINE_string("kl", "gamma", "KL type: gamma, mc, analytical, or cch")
flags.DEFINE_string("model_type", "negbio", "Model type: hnb, negbio, poisson, laplace, gaussian or categorical")
flags.DEFINE_string("dataset", "MNIST", "CIFAR16 or MNIST Omniglot svhn") 
flags.DEFINE_string("enc_type", "conv", "Choice of [linear, conv, mlp]")
flags.DEFINE_string("dec_type", "conv", "Choice of [linear, conv, mlp]")
flags.DEFINE_integer("seed", 42, "seed")
flags.DEFINE_integer("bsize", 512, "training batch size")
flags.DEFINE_integer("max_epochs", 200, "maximum epochs reached")
flags.DEFINE_integer("latent_dim", 256, "latent_dim")
flags.DEFINE_string("clf_type", "logreg", "Choice of [knn, logreg, svm]")
flags.DEFINE_bool("save_files", False, "if save npy files or not, default false")
flags.DEFINE_integer("mc_sample", 5, "# of samples for kl mc")
flags.DEFINE_float("tau", 1.0, "temperature")
flags.DEFINE_float("cch_tau", 0.1, "CCH CTS temperature (recommended: 0.05-0.2)")
flags.DEFINE_integer("cts_max_count", 64, "CCH CTS truncation M")
flags.DEFINE_bool("cch_detach_phi", False, "Detach reused CTS sample in CCH correction")
flags.DEFINE_string("hnb_groups_per_scale", "4,4", "HNB groups at each scale, top to bottom")
flags.DEFINE_string("hnb_channels_per_scale", "64,32", "HNB TD channels at each scale")
flags.DEFINE_string("hnb_latent_channels_per_scale", "16,16", "HNB latent channels at each scale")
flags.DEFINE_string("hnb_spatial_sizes", "", "Optional HNB spatial sizes, e.g. 7,14")
flags.DEFINE_float("hnb_gamma_rate_scale", 5.0, "Gamma-overlapping rate multiplier")
flags.DEFINE_float("hnb_free_bits", 0.0, "Free bits per HNB latent group")
flags.DEFINE_enum("hnb_kl_balance", "uniform", ["uniform", "scale"], "HNB group KL weighting")
flags.DEFINE_bool('kl_annealing', True, "kl annealing")  # in command line use --kl_annealing=False to stop auto kl annealing
flags.DEFINE_float('beta', 0.0, 'beta for kl')

flags.DEFINE_bool("local", False, "If local, run small set of MNIST")

def main(argv):
  
    seed_everything(FLAGS.seed, workers=True)



    with open(FLAGS.config_path, "r") as f:
            cfg = yaml.safe_load(f) 

    def update_cfg_from_flags(cfg, flags):
        def int_list(value):
            return [int(item.strip()) for item in value.split(',') if item.strip()]

        update_map = {
            ('model', 'reparam_type'): flags.reparam_type,
            ('dataset', 'name'): flags.dataset,
            ('model', 'name'): flags.model_type,
            ('model', 'kl'): flags.kl,
            ('model', 'latent_dim'): flags.latent_dim,
            ('encoder', 'latent_dim'): flags.latent_dim,
            ('decoder', 'latent_dim'): flags.latent_dim,
            ('logging', 'save_files'): flags.save_files,
            ('model', 'num_samples'): flags.mc_sample,
            ('encoder', 'type'): flags.enc_type,
            ('decoder', 'type'): flags.dec_type,
            ('model', 'beta'): flags.beta,
            ('model', 'kl_annealing'): flags.kl_annealing,
            ('model', 'tau'): flags.cch_tau if flags.kl == 'cch' else flags.tau,
            ('model', 'cts_max_count'): flags.cts_max_count,
            ('model', 'cch_detach_phi'): flags.cch_detach_phi,
            ('model', 'groups_per_scale'): int_list(flags.hnb_groups_per_scale),
            ('model', 'channels_per_scale'): int_list(flags.hnb_channels_per_scale),
            ('model', 'latent_channels_per_scale'): int_list(flags.hnb_latent_channels_per_scale),
            ('model', 'spatial_sizes'): (
                int_list(flags.hnb_spatial_sizes)
                if flags.hnb_spatial_sizes.strip() else None
            ),
            ('model', 'gamma_rate_scale'): flags.hnb_gamma_rate_scale,
            ('model', 'free_bits'): flags.hnb_free_bits,
            ('model', 'kl_balance'): flags.hnb_kl_balance,
        }

        for (section, key), value in update_map.items():
            cfg.setdefault(section, {})[key] = value

    update_cfg_from_flags(cfg, FLAGS)
    name = f'{FLAGS.model_type}-{FLAGS.reparam_type}-{FLAGS.kl}-{FLAGS.dataset}-{FLAGS.seed}'
    data_dir = "/Data/Datasets/" 
    project_name = FLAGS.model_type 
    root_dir = "ckpt/method-1/"

    checkpoint_dir = os.path.join(root_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True) 
    
    if FLAGS.model_type == "hnb" or cfg['encoder']['type'] == "conv":
        flatten_flag = False
    else:
        flatten_flag = True


    if FLAGS.local:
        # LOCAL IS USED FOR DEBUGGING PURPOSE ONLY - RUN ON LOCAL CPU ENV
        dm = DataModule(FLAGS.dataset, batch_size=FLAGS.bsize, flatten=flatten_flag, use_subset=True)
    else:
        dm = DataModule(FLAGS.dataset, batch_size=FLAGS.bsize, flatten=flatten_flag)
    model = VAETrainer(cfg)

    if FLAGS.local:
        accelerator = "cpu"
        devices = 1
        strategy = None
    else:
        accelerator = "gpu"
        devices = [0]
        if FLAGS.model_type == "gaussian":
            strategy = "ddp_find_unused_parameters_true"
        else:
            strategy = "ddp"

            
    trainer_args = {
    "callbacks": [
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor='val_elbo',
                save_top_k=1,
                mode='min',
                verbose=False
            ),
        ],
        # "logger": wandb.WandbLogger(project=project_name, name=name, save_code=False,offline=True),
        "logger": False,
        "gradient_clip_val": 1.0,
        "accelerator": accelerator,
        'devices':devices
    }
    if strategy is not None:
        trainer_args["strategy"] = strategy
    if trainer_args["logger"] and hasattr(trainer_args["logger"], "watch"):
        trainer_args["logger"].watch(model, log="all")

    
    trainer = pl.Trainer(
        **trainer_args,
        default_root_dir=checkpoint_dir,
        max_epochs=FLAGS.max_epochs,
        num_sanity_val_steps=0
    )


    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)



if __name__ == "__main__":
    app.run(main)
