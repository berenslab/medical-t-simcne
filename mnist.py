import numpy as np
import medmnist.dataset
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from evaluation.eval import knn_acc,silhouette_score_

root='datasets'
dataset_train = medmnist.dataset.DermaMNIST(root=root, split='train', transform=None, target_transform=None, download=True)
dataset_test = medmnist.dataset.DermaMNIST(root=root, split='test', transform=None, target_transform=None, download=True)
dataset_val = medmnist.dataset.DermaMNIST(root=root, split='val', transform=None, target_transform=None, download=True)
dataset_full = [dataset_train, dataset_test,dataset_val]

for dataset in dataset_full:
        dataset.labels = dataset.labels.squeeze()
dataset_full = ConcatDataset(dataset_full)

labels = np.array([lbl for img, lbl in dataset_full])


batch_size=1024
total_epochs=[1,1,1]

tsimcne = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs)


Y = tsimcne.fit_transform(dataset_full)

kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)
print(f"kNN_score: {kNN_score}")
print(f"Silhouette score: {sil_score}")









