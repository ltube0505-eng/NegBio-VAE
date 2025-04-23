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


FLAGS = flags.FLAGS
flags.DEFINE_string("reparam_type", "gamma", "Model type: gamma or gumbel")
flags.DEFINE_string("model_type", "negbio", "Model type: negbio or poisson")
flags.DEFINE_string("dataset", "CIFAR16", "dataset name") #CIFAR16 MNIST
flags.DEFINE_integer("seed", 42, "dataset name")
flags.DEFINE_bool("local", False, "If local, run small set of MNIST")


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

    cfg['datasetname'] = FLAGS.dataset
    if cfg['encoder']['type'] == "conv":
        flatten_flag = False
    else:
        flatten_flag = True

    print(flatten_flag)

    if FLAGS.local:
        dm = MNISTDataModule(data_dir='./Datasets', batch_size=bsize)
    else:
        dm = DataModule(FLAGS.dataset, batch_size=bsize, flatten=flatten_flag)
    
    
    model = VAETrainer(cfg)
    # print("🔍 Trainable parameters in the model:")
    # for name, param in model.named_parameters():
    #     if "logits" in name:
    #         print(f"  ✅ {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            
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
        
        "accelerator": "gpu",
        'devices':[2,3],
        'strategy':"ddp"
    }
    trainer_args["logger"].watch(model, log="all")


    trainer = pl.Trainer(
        **trainer_args,
        default_root_dir=checkpoint_dir,
        max_epochs=500,
        num_sanity_val_steps=0,
    )


    trainer.fit(model, datamodule=dm)
    



if __name__ == "__main__":
    app.run(main)