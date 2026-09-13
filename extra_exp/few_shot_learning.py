import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
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
#from analysis import train_clf_analysis
from sklearn.preprocessing import StandardScaler

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
flags.DEFINE_string("kl", "analytical", "type: mc or analytical")
flags.DEFINE_string("reparam_type", "gumbel", "Model type: gamma or gumbel")
flags.DEFINE_string("dataset", "CIFAR16", "Dataset name (CIFAR10, CelebA64, etc.)")
flags.DEFINE_integer("num_samples", 64, "Number of generated images")
flags.DEFINE_string("outdir", "./generated_eval", "Dir to save generated images")


@torch.no_grad()
def get_latents(lit_model, dataloader, device):
    lit_model.eval().to(device)
    validation = not torch.is_grad_enabled()
    dist_type = lit_model.model.dist_type
    Z = []
    Y = []
    m = lit_model.model
    m.latent_dim = 256
    latent_target_dim = 256
    for x,label in dataloader:
        x = x.to(device)
        if dist_type == "negbio":
            dist, (log_r_post, logit_p), z, y = m(x)
            z_repr = logit_p
        elif dist_type == "gaussian":
            dist, (loc, log_scale), z, y = m(x)
            z_repr = loc
        elif dist_type == "laplace":
            dist, (loc, log_scale), z, y = m(x)
            z_repr = loc
        elif dist_type == "categorical":
            dist, logit_p, z, y = m(x)
            z_repr = logit_p
        elif dist_type == 'poisson':
            dist, du, z, y = m(x)
            z_repr = du
        else:
            raise NotImplementedError            
        Z.append(z_repr.cpu())
        Y.append(label.cpu())
    Z = torch.cat(Z,dim=0).detach().numpy()
    Y = torch.cat(Y,dim=0).detach().numpy()
    assert Z.shape[1] == latent_target_dim, f"latent_dim != {m.latent_dim}, got {Z.shape}"
    return Z, Y

def few_shot_eval(Z_train, Y_train, Z_test, Y_test, clf_type="logreg", shots=5, repeats=5):
    classes = np.unique(Y_train)
    acc_list = []
    for _ in range(repeats):
        idxs = []
        for c in classes:
            c_idx = np.where(Y_train == c)[0]
            sampled = np.random.choice(c_idx, size=shots, replace=False)
            idxs.extend(sampled)
        idxs = np.array(idxs)

        Z_few = Z_train[idxs]
        Y_few = Y_train[idxs]

        if clf_type == "logreg":
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        elif clf_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=3)
        else:
            raise NotImplementedError

        clf.fit(Z_few, Y_few)
        y_pred = clf.predict(Z_test)
        acc = accuracy_score(Y_test, y_pred)
        acc_list.append(acc)

    return np.mean(acc_list), np.std(acc_list)


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
    # cfg["model"]["latent_dim"] = 100
    # cfg["encoder"]["latent_dim"] = 100
    # cfg["decoder"]["latent_dim"] = 100

    # ========= Load model =========
    print(f"Loading model from {FLAGS.ckpt} ...")
    lit_model = VAETrainer.load_from_checkpoint(FLAGS.ckpt, cfg=cfg, map_location=device,strict=False)
    lit_model.eval().to(device)
    latent_dim = cfg["model"]["latent_dim"]

    dm = DataModule(FLAGS.dataset, batch_size=64, flatten=False)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    Z_train, Y_train = get_latents(lit_model,train_loader,device)
    Z_test, Y_test = get_latents(lit_model,test_loader,device)

    # few-shot lorg
    mean_acc, std_acc = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="logreg", shots=1, repeats=10)
    print(f"1-shot logreg: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="logreg", shots=5, repeats=10)
    print(f"5-shot logreg: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="logreg", shots=10, repeats=10)
    print(f"10-shot logreg: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="logreg", shots=20, repeats=10)
    print(f"20-shot logreg: {mean_acc:.3f} ± {std_acc:.3f}")


    #few-shot knn
    mean_acc, std_acc = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="knn", shots=1, repeats=10)
    print(f"1-shot knn: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="knn", shots=5, repeats=10)
    print(f"5-shot knn: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="knn", shots=10, repeats=10)
    print(f"10-shot knn: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc  = few_shot_eval(Z_train, Y_train, Z_test, Y_test,clf_type="knn", shots=20, repeats=10)
    print(f"20-shot knn: {mean_acc:.3f} ± {std_acc:.3f}")

if __name__ == "__main__":
    app.run(main)
