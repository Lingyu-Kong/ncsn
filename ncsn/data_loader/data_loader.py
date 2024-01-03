import os
import torch
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from ncsn.data_loader.celeba import CelebA

def get_data_loader(
    dataset: str = "MNIST",
    data_folder: str = "./data",
    image_size: int = 32,
    batch_size: int = 64,
    random_flip: bool = False,
):
    if random_flip is False:
        train_transform = test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    if dataset == "MNIST":
        train_dataset = MNIST(
            root=data_folder, train=True, transform=train_transform, download=True
        )
        test_dataset = MNIST(
            root=data_folder, train=False, transform=test_transform, download=True
        )
    elif dataset == "CIFAR10":
        train_dataset = CIFAR10(
            root=data_folder, train=True, transform=train_transform, download=True
        )
        test_dataset = CIFAR10(
            root=data_folder, train=False, transform=test_transform, download=True
        )
    elif dataset == "SVHN":
        train_dataset = SVHN(
            root=data_folder, split="train", transform=train_transform, download=True
        )
        test_dataset = SVHN(
            root=data_folder, split="test", transform=test_transform, download=True
        )
    elif dataset == "CELEBA":
        train_dataset = CelebA(
            root=data_folder,
            split="train",
            transform=transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]), download=True
        )

        test_dataset = CelebA(
            root=data_folder,
            split="test",
            transform=transforms.Compose([
                transforms.CenterCrop(140),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]), download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, 
    )
    return train_dataloader, test_dataloader

if __name__=="__main__":
    train_dataloader, test_dataloader = get_data_loader(dataset="CIFAR10", image_size=32, random_flip=True)
    print(len(train_dataloader.dataset))
    print(len(test_dataloader.dataset))
    for i, (images, labels) in enumerate(train_dataloader):
        print(images.shape)
        print(labels.shape)
        break
    for i, (images, labels) in enumerate(test_dataloader):
        print(images.shape)
        print(labels.shape)
        break