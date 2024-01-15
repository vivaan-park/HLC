import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


def load_data(image_size):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = CelebA(root='data', split='train', transform=train_transform, download=True)
    val_dataset = CelebA(root='data', split='valid', transform=val_transform, download=True)
    test_dataset = CelebA(root='data', split='test', transform=val_transform, download=True)

    return train_dataset, val_dataset, test_dataset


def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       drop_last=False,
                                       shuffle=False)
