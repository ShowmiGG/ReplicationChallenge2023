
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np 
import os, shutil

def tiny_imagenet_dataloaders(data_dir, transform_train=True, AugMax=None, **AugMax_args):
    
    train_root = os.path.join(data_dir, 'train')
    validation_root = os.path.join(data_dir, 'val/images')
    print('Training images loading from %s' % train_root)
    print('Validation images loading from %s' % validation_root)
    
    if AugMax is not None:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)]) if transform_train else None
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
        train_data = AugMax(train_data, test_transform, 
            mixture_width=AugMax_args['mixture_width'], mixture_depth=AugMax_args['mixture_depth'], aug_severity=AugMax_args['aug_severity'], 
        )

    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor()]) if transform_train else transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
    return train_data, test_data

