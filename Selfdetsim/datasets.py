import os
import random
import json
from scipy.io import loadmat
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset

from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation,RandomCrop16
from torchvision import transforms as T
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, ImageNet, Caltech101, Caltech256

import kornia.augmentation as K

class ImageList(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('/home/gpu/xxxxxx/imagenet100.txt') as f: #imagenet100 data
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

class Pets(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1])-1
            samples.append((path, label))

        super().__init__(samples, transform)



def load_pretrain_datasets(dataset='cifar10',
                           datadir='/data',
                           color_aug='default'):

    if dataset == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        train_transform = MultiView(RandomResizedCrop(224, scale=(0.2, 1.0)))

        test_transform = T.Compose([T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           #K.RandomResizedCrop(224, scale=(0.2, 1.0)),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           #K.RandomResizedCrop(224, scale=(0.2, 1.0)),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t3 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           #K.RandomResizedCrop(224, scale=(0.2, 1.0)),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),                           
                           K.Normalize(mean, std))
        t4 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           #K.RandomResizedCrop(224, scale=(0.2, 1.0)),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(23, (0.1, 2.0)),                           
                           K.Normalize(mean, std))
        trainset = ImageNet100(datadir, split='train', transform=train_transform)
        valset   = ImageNet100(datadir, split='train', transform=test_transform)
        testset  = ImageNet100(datadir, split='val', transform=test_transform)


        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    elif dataset == 'stl10_sol':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        train_transform = MultiView(RandomResizedCrop(96, scale=(0.2, 1.0)))

        test_transform = T.Compose([T.Resize(96),
                                    T.CenterCrop(96),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        t1 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))
        t2 = nn.Sequential(K.RandomHorizontalFlip(),
                           ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                           K.RandomSolarize(0.5, 0.0, p=0.5),
                           K.RandomGrayscale(p=0.2),
                           GaussianBlur(9, (0.1, 2.0)),
                           K.Normalize(mean, std))

        trainset = STL10(datadir, split='train+unlabeled', transform=train_transform)
        valset   = STL10(datadir, split='train',           transform=test_transform)
        testset  = STL10(datadir, split='test',            transform=test_transform)

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(train=trainset,
                val=valset,
                test=testset,
                t1=t1, t2=t2,t3=t3,t4=t4)

def load_datasets(dataset='cifar10',
                  datadir='/data',
                  pretrain_data='stl10'):

    if pretrain_data == 'imagenet100':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    elif pretrain_data == 'stl10':
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        transform = T.Compose([T.Resize(96, interpolation=Image.BICUBIC),
                               T.CenterCrop(96),
                               T.ToTensor(),
                               T.Normalize(mean, std)])

    generator = lambda seed: torch.Generator().manual_seed(seed)
    if dataset == 'imagenet100':
        trainval = ImageNet100(datadir, split='train', transform=transform)
        train, val = None, None
        test     = ImageNet100(datadir, split='val', transform=transform)
        num_classes = 100

    elif dataset == 'food101':
        trainval   = Food101(root=datadir, split='train', transform=transform)
        train, val = random_split(trainval, [7575, 68175], generator=generator(42))
        test       = Food101(root=datadir, split='test',  transform=transform)
        num_classes = 101

    elif dataset == 'cifar10':
        trainval   = CIFAR10(root=datadir, train=True, download=True, transform=transform)
        train, val = random_split(trainval, [45000, 5000], generator=generator(43))
        test       = CIFAR10(root=datadir, train=False, transform=transform)
        num_classes = 10

    elif dataset == 'cifar100':
        trainval   = CIFAR100(root=datadir, train=True, download=True, transform=transform)
        train, val = random_split(trainval, [45000, 5000], generator=generator(44))
        test       = CIFAR100(root=datadir, train=False, transform=transform)
        num_classes = 100

    elif dataset == 'sun397':
        trn_indices, val_indices = torch.load('splits/sun397.pth')
        trainval = SUN397(root=datadir, split='Training', transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = SUN397(root=datadir, split='Testing',  transform=transform)
        num_classes = 397

    elif dataset == 'dtd':
        train    = DTD(root=datadir, split='train', transform=transform)
        val      = DTD(root=datadir, split='val',   transform=transform)
        trainval = ConcatDataset([train, val])
        test     = DTD(root=datadir, split='test',  transform=transform)
        num_classes = 47

    elif dataset == 'pets':
        trainval   = Pets(root=datadir, split='trainval', transform=transform)
        train, val = random_split(trainval, [2940, 740], generator=generator(49))
        test       = Pets(root=datadir, split='test',     transform=transform)
        num_classes = 37

    elif dataset == 'caltech101':
        transform.transforms.insert(0, T.Lambda(lambda img: img.convert('RGB')))
        D = Caltech101(datadir, transform=transform)
        trn_indices, val_indices, tst_indices = torch.load('splits/caltech101.pth')
        train    = Subset(D, trn_indices)
        val      = Subset(D, val_indices)
        trainval = ConcatDataset([train, val])
        test     = Subset(D, tst_indices)
        num_classes = 101

   
    elif dataset == 'stl10':
        trainval   = STL10(root=datadir, split='train', transform=transform)
        test       = STL10(root=datadir, split='test',  transform=transform)
        train, val = random_split(trainval, [4500, 500], generator=generator(50))
        num_classes = 10

    elif dataset == 'mit67':
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        train, val = random_split(trainval, [4690, 670], generator=generator(51))
        num_classes = 67

    elif dataset == 'cub200':
        trn_indices, val_indices = torch.load('splits/cub200.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 200

    elif dataset == 'dog':
        trn_indices, val_indices = torch.load('splits/dog.pth')
        trainval = ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        train    = Subset(trainval, trn_indices)
        val      = Subset(trainval, val_indices)
        test     = ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
        num_classes = 120

    return dict(trainval=trainval,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)



