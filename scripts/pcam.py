import os
import torch
import h5py
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from tsimcne.evaluation.eval import knn_acc,silhouette_score_
from dataloader.pcam import PatchCamelyon


batch_size=512
total_epochs=[1000,50,450]

data_folder='../datasets/pcam16/'
train_ds =PatchCamelyon(data_folder,mode='train',download=True) #change to `download=False` if you have downloaded it
test_ds=PatchCamelyon(data_folder,mode='test',download=True)
val_ds=PatchCamelyon(data_folder,mode='valid',download=True)


dataset_full = torch.utils.data.ConcatDataset([train_ds, test_ds,val_ds])
labels = np.array([lbl for img, lbl in dataset_full])

tsimcne = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs,
                   seed=1011 
                   )


Y = tsimcne.fit_transform(dataset_full)

kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)
print(f"kNN_score: {kNN_score}")
print(f"Silhouette score: {sil_score}")

fig, ax = plt.subplots()
ax.scatter(*Y.T, c=labels)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}", fontsize=7)
fig.savefig("tsimcne.png")
