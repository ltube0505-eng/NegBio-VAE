import math
import os

import torch
from PIL import ImageOps,Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split, Dataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class CelebAHQDataset(Dataset):
    def __init__(self, root, transform=None, subsample_size=None):
        super().__init__()
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform

        self.images = []
        modes = ["train", "val"]
        subfolders = ["male", "female"]

        for mode in modes:
            for folder in subfolders:
                img_path = os.path.join(self.root, mode, folder)
                for img in sorted(os.listdir(img_path)):
                    self.images.append(os.path.join(img_path, img))

        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img,idx

    def __len__(self):
        return len(self.images)

class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_transform(dataset_name, device, grey=False, augment=False, flatten=True, resize=None, crop64=False):
    tf = []

    if dataset_name == 'Omniglot':
        tf.append(transforms.Resize(28, interpolation=InterpolationMode.NEAREST))
        tf.append(transforms.Lambda(lambda img: ImageOps.invert(img.convert('L'))))
    if dataset_name == "CIFAR16":
        tf.append(transforms.Resize(16)) 
        
    if dataset_name == "CelebAHQ":
        tf.append(transforms.Resize(128))

    if crop64:
        tf.append(CropCelebA64())

    if resize is not None:
        tf.append(transforms.Resize(resize))

    if grey:
        tf.append(transforms.Grayscale())

    if augment and dataset_name in ['CIFAR10', 'CIFAR16']:
        tf.insert(0, transforms.RandomHorizontalFlip(p=0.5))

    tf.append(transforms.ToTensor())

    if dataset_name in ['SVHN', 'CIFAR10', 'CelebA','CIFAR16', 'CelebA64', 'CelebAHQ', 'FFHQ']:
        if grey:
            tf.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            tf.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3))

    if flatten:
        tf.append(transforms.Lambda(lambda x: x.view(-1)))

    tf.append(transforms.Lambda(lambda x: x.to(device)))
    return transforms.Compose(tf)




def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


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
        if name == "fmnist":
            datasets.FashionMNIST(root=self.data_dir,train=True,download=True)
            datasets.FashionMNIST(root=self.data_dir,train=False,download=True)
        elif name == "svhn":
            print("Prepare SVHN...")
            datasets.SVHN(root=self.data_dir, split='train', download=True)
            datasets.SVHN(root=self.data_dir, split='test', download=True)
        elif name == "omniglot":
            print("Prepare Omniglot...")
            datasets.Omniglot(root=self.data_dir, background=True, download=True)
            datasets.Omniglot(root=self.data_dir, background=False, download=True)
        elif name in ["celeba", "celeba64"]:
            print("Prepare Celeba")
            datasets.CelebA(root=self.data_dir,split='train',download=False)
            datasets.CelebA(root=self.data_dir,split='valid',download=False)
            datasets.CelebA(root=self.data_dir,split='test',download=False)
        elif name in ["celebahq"]:
            print("prepare CelebAHQ")
        else:
            if name == "cifar16":
                self.dataset_name = "CIFAR10"
                print("Prepare CIFAR16...")
            dataset_cls = getattr(datasets, self.dataset_name)
            dataset_cls(root=self.data_dir, train=True, download=True)
            dataset_cls(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None): 
        name = self.dataset_name
        
        if name == "fmnist":
            full = datasets.FashionMNIST(root=self.data_dir,train=True,transform=self.transform)
            self.test_set = datasets.FashionMNIST(root=self.data_dir,train=False,transform=self.transform)

        elif name == "SVHN":
            full = datasets.SVHN(root=self.data_dir, split="train", transform=self.transform)
            self.test_set = datasets.SVHN(root=self.data_dir, split="test", transform=self.transform)
        elif name == "Omniglot":
            full = datasets.Omniglot(root=self.data_dir, background=True, transform=self.transform)
            self.test_set = datasets.Omniglot(root=self.data_dir, background=False, transform=self.transform)
        elif name == "CelebA64":
            # train_transform, valid_transform = _data_transforms_celeba64(resize=64)
            full = datasets.CelebA(root=self.data_dir, split="train",
                                   transform=get_transform(name, self.device, self.grey, self.augment, self.flatten, resize=64, crop64=True))
            self.valid_set = datasets.CelebA(root=self.data_dir, split="valid", 
                                            transform=get_transform(name, self.device, self.grey, False, self.flatten, resize=64, crop64=True))
            self.test_set = datasets.CelebA(root=self.data_dir, split="test",
                                            transform=get_transform(name, self.device, self.grey, False, self.flatten, resize=64, crop64=True))

        elif name == "CelebAHQ":
            full = CelebAHQDataset(root=os.path.join(self.data_dir, "celeba_hq"), transform=self.transform)
            self.test_set = None
        elif name == "FFHQ":
            full = ImageFolder(root=os.path.join(self.data_dir, "ffhq/train"),
                               transform=get_transform(name, self.device, self.grey, self.augment, self.flatten, resize=256))
            self.test_set = ImageFolder(root=os.path.join(self.data_dir, "ffhq/test"),
                                        transform=get_transform(name, self.device, self.grey, False, self.flatten, resize=256))
        else:
            if name == "CIFAR16":
                dataset_cls = getattr(datasets, "CIFAR10")
            else:
                dataset_cls = getattr(datasets, name)
            full = dataset_cls(root=self.data_dir, train=True, transform=self.transform)
            self.test_set = dataset_cls(root=self.data_dir, train=False, transform=self.transform)



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
            if name == 'CelebA64':
                self.train_set, self.val_set = full, self.valid_set
            elif name == "CelebAHQ":
                total_len = len(full)
                train_len = int (0.8 * total_len)
                val_len = int(0.1 * total_len)
                test_len = total_len - train_len - val_len
                self.train_set, self.val_set, self.test_set = random_split(full,[train_len,val_len,test_len], 
                generator=torch.Generator().manual_seed(42)) 

            else:
                self.train_set, self.val_set = random_split(full, [train_len, val_len],
                generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def fid_dataloader(self):
        if self.dataset_name in ['CIFAR16','CIFAR10']:
            self.fid_set = ConcatDataset([self.train_set, self.val_set])
        elif self.dataset_name == 'CelebA64':
            self.fid_set = ConcatDataset([self.train_set, self.val_set, self.test_set])
        elif self.dataset_name == 'SVHN':
            self.fid_set = ConcatDataset([self.train_set, self.val_set])
        elif self.dataset_name == 'CelebAHQ':
            self.fid_set = ConcatDataset([self.train_set, self.val_set, self.test_set])
        elif self.dataset_name == 'MNIST':
            self.fid_set = ConcatDataset([self.train_set, self.val_set])
        elif self.dataset_name == 'fmnist':
            self.fid_set = ConcatDataset([self.train_set, self.val_set])
        print(len(self.fid_set))
        return DataLoader(self.fid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



