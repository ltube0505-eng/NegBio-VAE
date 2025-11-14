import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from sympy import imageset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from data import DataModule
from absl import app, flags
from vaetrainer import VAETrainer
import yaml
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from distribution import Categorical, Gaussian, Laplace, NegBinomial, Poisson
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
import math
import random
# from analysis import train_clf_analysis
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




device = "cuda" if torch.cuda.is_available() else "cpu"
FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt", None, "Path to checkpoint")
flags.DEFINE_string("config_path", "configs/train_config.yaml", "Path to config yaml")
flags.DEFINE_string("model_type", "negbio", "Model type: negbio, poisson, laplace, gaussian or categorical")
flags.DEFINE_string("kl", "gamma", "type: mc or analytical")
flags.DEFINE_string("reparam_type", "gumbel", "Model type: gamma or gumbel")
flags.DEFINE_string("dataset", "CIFAR16", "Dataset name (CIFAR10, CelebA64, etc.)")
flags.DEFINE_integer("num_samples", 64, "Number of generated images")
flags.DEFINE_string("outdir", "./generated_eval", "Dir to save generated images")
flags.DEFINE_integer("mc_sample", 5, "# of samples for kl mc")
flags.DEFINE_string("enc_type", "conv", "Choice of [linear, conv, mlp]")
flags.DEFINE_string("dec_type", "conv", "Choice of [linear, conv, mlp]")
flags.DEFINE_float('beta', 0.0, 'beta for kl')


@torch.no_grad()
def test_recon_loss(model, data_loader, device='cuda', data_range=1.0):
    model.eval().to(device)
    m = model.model
    dist_type = m.dist_type

    mse_sum = 0.0          
    pixel_count = 0        
    ssim_sum = 0.0         
    n_images = 0           
    sample_count = 0 

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    for x, label in data_loader:
        x = x.to(device).float()
        if dist_type == "negbio":
            dist, lp, z, y = m(x)
        elif dist_type == "gaussian":
            dist, (loc, log_scale), z, y = m(x)
        elif dist_type == "laplace":
            dist, (loc, log_scale), z, y = m(x)
        elif dist_type == "categorical":
            dist, logit_p, z, y = m(x)
        elif dist_type == 'poisson':
            dist, du, z, y = m(x)
        else:
            raise NotImplementedError(dist_type)

        if y.dim() == 2 and x.dim() > 2:
            y = y.view_as(x)
        


        mse_sum += torch.nn.functional.mse_loss(y, x, reduction='sum').item()
        pixel_count += x.numel()
        x01 = y #(y+1)/2  # mnist fashion-mnist don't need  (x+1)/2  Cifar CelebA need
        y01 = x #(x+1)/2
        ssim_batch_mean = ssim_metric(x01, y01).item()   
        ssim_sum += ssim_batch_mean * x.size(0)
        n_images += x.size(0)

        sample_count += x.size(0)


    mse_mean = mse_sum / pixel_count
    ssim_mean = ssim_sum / n_images

    return mse_mean, ssim_mean



def main(argrv):
    assert FLAGS.ckpt is not None, "Please provide --ckpt checkpoint path"
    os.makedirs(FLAGS.outdir, exist_ok=True)

    # ========= Load config =========
    with open(FLAGS.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["name"] = FLAGS.dataset
    cfg["model"]["name"] = FLAGS.model_type
    cfg["model"]['kl'] = FLAGS.kl
    cfg["model"]['reparam_type'] = FLAGS.reparam_type
    cfg["model"]["num_samples"] = FLAGS.mc_sample
    cfg["encoder"]["type"] = FLAGS.enc_type
    cfg["decoder"]["type"] = FLAGS.dec_type
    cfg["model"]["beta"] = FLAGS.beta

    # cfg["model"]["latent_dim"] = 512
    # cfg["encoder"]["latent_dim"] = 512
    # cfg["decoder"]["latent_dim"] = 512

    # ========= Load model =========
    print(f"Loading model from {FLAGS.ckpt} ...")
    lit_model = VAETrainer.load_from_checkpoint(FLAGS.ckpt, cfg=cfg, map_location=device,strict=False)
    lit_model.eval().to(device)
    latent_dim = cfg["model"]["latent_dim"]
    print(cfg["model"]["name"])
    print(cfg["encoder"]["type"])
    print(cfg["decoder"]["type"])
    print(cfg["model"]["reparam_type"])

    dm = DataModule(FLAGS.dataset, data_dir='/root/nbvae/datasets',batch_size=100, flatten=False)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    res = test_recon_loss(lit_model,test_loader,device='cuda')
    print(f"Reconstruction MSE: {res[0]:.6f}, SSIM: {res[1]:.6f}")




if __name__ == "__main__":
    app.run(main)
