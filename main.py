from data import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from absl import app, flags
import warnings 
import torch
import os
import pytorch_lightning as pl
import wandb as wdb
from pytorch_lightning.loggers import wandb
import yaml
from vaetrainer import VAETrainer
warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "MNIST", "dataset name")


def main(argv):
    # version 1 (download)
    name = 'PVAE_minimal' #change to your run name
    data_dir = "/Data/Datasets/" #change to your data directory
    project_name = "PVAE" #change to your wandb project name
    root_dir = "data"
    bsize = 256
    # train_device = "0"
    # device = train_device + "," #lightning device formatting
    checkpoint_dir = os.path.join(root_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dm = MNISTDataModule(data_dir='./Datasets', batch_size=16)
    torch.autograd.set_detect_anomaly(True)
    # version 2 (local)
    # Make sure the local .npy files exist in data_dir!
    # dm = LocalMNISTDataModule(data_dir='./local_mnist', batch_size=64)
    dataset = "MNIST"
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    
    model = VAETrainer(cfg)
    trainer_args = {
    "callbacks": [
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor='val_elbo',
                save_top_k=1,
                mode='min',
                verbose=False
            )
        ],
        "accelerator": "cpu",
        "logger": wandb.WandbLogger(project=project_name, name=name, save_code=False),
        "gradient_clip_val": 1.0,
    }
    trainer_args["logger"].watch(model, log="all")


    trainer = pl.Trainer(
        **trainer_args,
        default_root_dir=checkpoint_dir,
        max_epochs=500,
        num_sanity_val_steps=0
    )


    trainer.fit(model, datamodule=dm)
    



if __name__ == "__main__":
    app.run(main)