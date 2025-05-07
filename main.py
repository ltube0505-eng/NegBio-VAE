import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from absl import app, flags
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, wandb

import wandb as wdb
from callback import GumbelMonitorCallback
from data import DataModule, MNISTDataModule
from vaetrainer import VAETrainer

warnings.filterwarnings("ignore")
GPU = '1' #设置GPU 0 1 2 可见
os.environ['CUDA_VISIBLE_DEVICES'] =GPU

FLAGS = flags.FLAGS
flags.DEFINE_string("reparam_type", "gumbel", "Model type: gamma or gumbel")
flags.DEFINE_string("kl", "gamma", "type: mc or analytical")
flags.DEFINE_string("model_type", "negbio", "Model type: negbio, poisson, laplace, gaussian or category")
flags.DEFINE_string("dataset", "MNIST", "CIFAR16 or MNIST Omniglot svhn") #
flags.DEFINE_integer("seed", 42, "dataset name")
flags.DEFINE_bool("local", False, "If local, run small set of MNIST")
flags.DEFINE_integer("bsize_local", 64, "training batch size for local")
flags.DEFINE_integer("max_epochs", 200, "maximum epochs reached")
flags.DEFINE_integer("latent_dim", 10, "latent_dim")

def main(argv):
  
    seed_everything(FLAGS.seed, workers=True)

    name = f'{FLAGS.model_type}-{FLAGS.reparam_type}-{FLAGS.dataset}-{FLAGS.seed}'
    data_dir = "/Data/Datasets/" 
    project_name = "negbio" 
    root_dir = "data"
    bsize = 512

    checkpoint_dir = os.path.join(root_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if FLAGS.model_type == "poisson":
        with open("configs/pvaeconfig.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    else:
        if FLAGS.reparam_type == "gamma":
            with open("configs/gammaconfig.yaml", "r") as f:
                cfg = yaml.safe_load(f)
        elif FLAGS.reparam_type =="gumbel":
            with open("configs/gumbelconfig.yaml", "r") as f:
                cfg = yaml.safe_load(f)


    cfg['dataset']['name'] = FLAGS.dataset
    cfg['model']['name'] = FLAGS.model_type
    cfg['model']['kl'] = FLAGS.kl
    cfg['model']['latent_dim'] = FLAGS.latent_dim
    cfg['encoder']['latent_dim'] = FLAGS.latent_dim
    cfg['decoder']['latent_dim'] = FLAGS.latent_dim
    

    
    if cfg['encoder']['type'] == "conv":
        flatten_flag = False
    else:
        flatten_flag = True

    print(flatten_flag)

    if FLAGS.local:
        dm = DataModule(FLAGS.dataset, batch_size=bsize, flatten=flatten_flag, use_subset=True)
    else:
        dm = DataModule(FLAGS.dataset, batch_size=bsize, flatten=flatten_flag)
    
    
    model = VAETrainer(cfg)
    # print("🔍 Trainable parameters in the model:")
    # for name, param in model.named_parameters():
    #     if "logits" in name:
    #         print(f"  ✅ {name}: requires_grad={param.requires_grad}, shape={param.shape}")

    if FLAGS.local:
        accelerator = "cpu"
        devices = 1
        strategy = None
    else:
        accelerator = "gpu"
        devices = [0]
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
            GumbelMonitorCallback(log_every_n_steps=1)

        ],
        #"accelerator": "auto",
        "logger": wandb.WandbLogger(project=project_name, name=name, save_code=False),
        "gradient_clip_val": 1.0,
        "accelerator": accelerator,
        'devices':devices
        # 'strategy':"ddp" if accelerator == "gpu" else None
    }
    if strategy is not None:
        trainer_args["strategy"] = strategy
    trainer_args["logger"].watch(model, log="all")

    
    trainer = pl.Trainer(
        **trainer_args,
        default_root_dir=checkpoint_dir,
        max_epochs=FLAGS.max_epochs,
        num_sanity_val_steps=0,
    )


    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    



if __name__ == "__main__":
    app.run(main)