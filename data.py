from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
import torch

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
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)