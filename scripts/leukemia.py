import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from tsimcne.evaluation.eval import knn_acc,silhouette_score_

class Leukemia(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_aml = os.path.join(self.image_folder, self.data.iloc[idx]['img'])
        label = self.data.iloc[idx]['labels']
        
        image = Image.open(image_aml).convert("RGB").resize((28,28))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

csv_file = '../datasets/train.csv'
image_folder = '../datasets/aml/images'

batch_size=1
total_epochs=[1,1,1]
dataset = Leukemia(csv_file, image_folder, transform=None)
labels = np.array([lbl for img, lbl in dataset])
tsimcne = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs,
                   )


Y = tsimcne.fit_transform(dataset)

kNN_score=knn_acc(Y,labels)
sil_score=silhouette_score_(Y,labels)
print(f"kNN_score: {kNN_score}")
print(f"Silhouette score: {sil_score}")

fig, ax = plt.subplots()
ax.scatter(*Y.T, c=labels)
ax.set_title(f"$k$NN acc. = {kNN_score}% sil score = {sil_score}", fontsize=7)
fig.savefig("../figures/tsimcne.png")

