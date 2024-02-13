from sklearn import neighbors, model_selection
from sklearn.metrics import silhouette_score
def knn_acc(Y,labels):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(Y,labels, 
                                                                        test_size=0.1, 
                                                                        random_state=0000,
                                                                        stratify=labels)

    knn = neighbors.KNeighborsClassifier(15)
    knn.fit(X_train, y_train)
    kNN_score=round((knn.score(X_test, y_test)*100),1)
    return kNN_score

def silhouette_score_(Y,labels):
    return silhouette_score(Y, labels).round(3)