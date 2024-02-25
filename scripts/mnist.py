import numpy as np
import sys
sys.path.append('..')
import medmnist.dataset
import matplotlib.pyplot as plt
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from tsimcne.evaluation.eval import knn_acc,silhouette_score_
from torch.utils.data import ConcatDataset
from tsimcne.dataset.average_border_pixels import avg_border_col_agg

root='../datasets'
dataset_train = medmnist.dataset.BloodMNIST(root=root, split='train', transform=None, target_transform=None, download=True)
dataset_test = medmnist.dataset.BloodMNIST(root=root, split='test', transform=None, target_transform=None, download=True)
dataset_val = medmnist.dataset.BloodMNIST(root=root, split='val', transform=None, target_transform=None, download=True)
dataset_full = [dataset_train, dataset_test,dataset_val]

for dataset in dataset_full:
        dataset.labels = dataset.labels.squeeze()
dataset_full = ConcatDataset(dataset_full)

labels = np.array([lbl for img, lbl in dataset_full])
border_pixels=avg_border_col_agg(dataset_full)


batch_size=1024
total_epochs=[1000,50,450]
crop_scale = 0.2, 1
size=28
data_aug=transforms.Compose(
            [
                transforms.RandomApply([
                       lambda img: transforms.functional.rotate(img,90)
             ]),
                transforms.RandomRotation(45,fill=border_pixels),
                transforms.RandomResizedCrop(size=size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4,contrast=0.4, 
                        saturation=0.4,hue=0.1 )
                    ], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
            ])


tsimcne = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs,
                   data_transform=data_aug, #add custom data augmentations
                   seed=1011 #set seed for reproducability
                   )


Y = tsimcne.fit_transform(dataset_full)

kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)
print(f"kNN_score: {kNN_score}")
print(f"Silhouette score: {sil_score}")

fig, ax = plt.subplots()
ax.scatter(*Y.T, c=labels)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}", fontsize=7)
fig.savefig("../figures/tsimcne.png")







