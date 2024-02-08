import numpy as np
import medmnist.dataset
import torch
from tsimcne.imagedistortions import *
from tsimcne.tsimcne import TSimCNE
from sklearn import neighbors, model_selection


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
total_epochs=[1000,50,450]

tsimcne_ = TSimCNE(batch_size=batch_size,
                   total_epochs=total_epochs)


Y = tsimcne_.fit_transform(dataset_full)


X_train, X_test, y_train, y_test = model_selection.train_test_split(Y,labels, 
                                                                    test_size=0.1, 
                                                                    random_state=4444,
                                                                    stratify=labels)

knn = neighbors.KNeighborsClassifier(15)
knn.fit(X_train, y_train)
kNN_score=knn.score(X_test, y_test).round(3)
print(f"kNN_score: {kNN_score}")








