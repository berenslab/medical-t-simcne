import numpy as np
import medmnist.dataset
import torch
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE


root='datasets'
dataset_train = medmnist.dataset.DermaMNIST(root=root, split='train', transform=None, target_transform=None, download=True)
dataset_test = medmnist.dataset.DermaMNIST(root=root, split='test', transform=None, target_transform=None, download=True)
dataset_val = medmnist.dataset.DermaMNIST(root=root, split='val', transform=None, target_transform=None, download=True)
dataset_full = torch.utils.data.ConcatDataset([dataset_train, dataset_test,dataset_val])
labels = np.array([lbl for img, lbl in dataset_full])


dataset_name = dataset_train.info['python_class']


size=28
crop_scale = 0.2, 1
batch_size=1024


augmentations=transforms.Compose(
            [
                transforms.RandomApply([
                    lambda img: transforms.functional.rotate(img,90)
               ]),
                transforms.RandomResizedCrop(size=size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4,contrast=0.4, 
                        saturation=0.4,hue=0.1 )
                    ], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])


epoch_total=[450,50,200]

tsimcne_ = TSimCNE(batch_size=batch_size,
                   total_epochs=epoch_total,
                   data_transform_train=augmentations)


Y = tsimcne_.fit_transform(dataset_full)











