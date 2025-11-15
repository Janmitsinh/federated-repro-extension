# src/datasets.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(data_root='./data', batch_size=256, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10(root=os.path.join(data_root, 'cifar10'), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=os.path.join(data_root, 'cifar10'), train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, test_loader
