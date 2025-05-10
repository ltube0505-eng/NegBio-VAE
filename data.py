import math
import os

import torch
from PIL import ImageOps
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './Datasets', batch_size: int = 16, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # Download MNIST once
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Load datasets
        full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        full_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

        # Randomly sample 100 training examples
        torch.manual_seed(0)
        train_idx = torch.randperm(len(full_train))[:200]
        self.mnist_train = Subset(full_train, train_idx)

        # Use 25 images for validation
        val_idx = torch.randperm(len(full_test))[:55]
        self.mnist_val = Subset(full_test, val_idx)

        # Use full test set
        test_idx = torch.randperm(len(full_test))[:25]
        self.mnist_test = Subset(full_test, test_idx)
        # self.mnist_test = full_test
        import math
        self.train_steps_per_epoch = math.ceil(len(self.mnist_train) / self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    




def get_transform(dataset_name, device, grey=False, augment=False, flatten=True):
    tf = []

    if dataset_name == 'Omniglot':
        tf.append(transforms.Resize(28, interpolation=InterpolationMode.NEAREST))
        tf.append(transforms.Lambda(lambda img: ImageOps.invert(img.convert('L'))))
    if dataset_name == "CIFAR16":
        tf.append(transforms.Resize(16))
    if grey:
        tf.append(transforms.Grayscale())

    if augment and dataset_name in ['CIFAR10', 'CIFAR16']:
        tf.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    tf.append(transforms.ToTensor())

    if dataset_name in ['SVHN', 'CIFAR10', 'CelebA','CIFAR16']:
        if grey:
            tf.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            tf.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)) #[-1,1] 

    if flatten:
        tf.append(transforms.Lambda(lambda x: x.view(-1)))

    tf.append(transforms.Lambda(lambda x: x.to(device)))
    return transforms.Compose(tf)


class DataModule(LightningDataModule):
    def __init__(self, 
                 dataset_name="MNIST", 
                 data_dir="./datasets", 
                 batch_size=64, 
                 num_workers=0,
                 val_split=0.1, 
                 device='cpu', 
                 grey = False, 
                 augment = False,
                 flatten = True,
                 use_subset=False,
                 train_subset_size=500,
                 val_subset_size=50,
                 seed = 1975):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = device
        self.grey = grey
        self.augment = augment
        self.num_workers = num_workers
        self.flatten = flatten
        self.use_subset = use_subset
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.seed = seed

        self.transform = get_transform(dataset_name, device, grey, augment, flatten)

    def prepare_data(self):
        name = self.dataset_name.lower()
        if name == "svhn":
            print("Prepare SVHN...")
            datasets.SVHN(root=self.data_dir, split='train', download=True)
            datasets.SVHN(root=self.data_dir, split='test', download=True)
        elif name == "omniglot":
            print("Prepare Omniglot...")
            datasets.Omniglot(root=self.data_dir, background=True, download=True)
            datasets.Omniglot(root=self.data_dir, background=False, download=True)
        else:
            if name == "cifar16":
                self.dataset_name = "CIFAR10"
                print("Prepare CIFAR16...")
            dataset_cls = getattr(datasets, self.dataset_name)
            dataset_cls(root=self.data_dir, train=True, download=True)
            dataset_cls(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        name = self.dataset_name

        
        if name == "SVHN":
            full = datasets.SVHN(root=self.data_dir, split="train", transform=self.transform)
            self.test_set = datasets.SVHN(root=self.data_dir, split="test", transform=self.transform)
        elif name == "Omniglot":
            full = datasets.Omniglot(root=self.data_dir, background=True, transform=self.transform)
            self.test_set = datasets.Omniglot(root=self.data_dir, background=False, transform=self.transform)
        else:
            if name == "CIFAR16":
                dataset_cls = getattr(datasets, "CIFAR10")
            else:
                dataset_cls = getattr(datasets, name)
            full = dataset_cls(root=self.data_dir, train=True, transform=self.transform)
            self.test_set = dataset_cls(root=self.data_dir, train=False, transform=self.transform)
        print("Dataset {} is set!".format(name))


        if self.use_subset:
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_idx = torch.randperm(len(full), generator=g)[:self.train_subset_size]
            val_idx = torch.randperm(len(full), generator=g)[self.train_subset_size:self.train_subset_size+self.val_subset_size]
          
            self.train_set = Subset(full, train_idx)
            self.val_set = Subset(full, val_idx)
            self.train_steps_per_epoch = math.ceil(len(self.train_set) / self.batch_size)

        else:
            val_len = int(len(full) * self.val_split)
            train_len = len(full) - val_len
            
            self.train_steps_per_epoch = math.ceil(train_len/ self.batch_size)

            self.train_set, self.val_set = random_split(full, [train_len, val_len])

        print("Train set size:", len(self.train_set))
        print("Val set size:", len(self.val_set))
        print("Test set size:", len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
