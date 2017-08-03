import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import linear_model, datasets
from sklearn import grid_search
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
#%%
def ft_map(df):
    df["DistFromCenter"] = np.sqrt(df["XCoord"]**2 + df["YCoord"]**2)
    class_mappling ={label:idx for idx,label in enumerate(np.unique(df["Competitor"]))}
    df["Competitor"] = df["Competitor"].map(class_mappling)
    return df
#%%
train_dataset = pd.read_csv("/home/lanhpham/python_code/MLPB-master/Problems/Classify Dart Throwers/_Data/train.csv")
train_dataset = ft_map(train_dataset)
#%%
test_dataset = pd.read_csv("/home/lanhpham/python_code/MLPB-master/Problems/Classify Dart Throwers/_Data/test.csv")
test_dataset = ft_map(test_dataset)
#%%
"""Split data_train into five folder"""
data_train_split = np.array_split(train_dataset,5)
testFold = []
trainFold = []
for i in range(0,5):
    testFold.append(data_train_split[i])
    train = pd.concat(data_train_split)
    train = train[~train["ID"].isin(data_train_split[i]["ID"])]
    trainFold.append(train)
#%%

#%%
"""Model KNN & SVM with  five folder"""
y_pred_KNN = []
y_pred_SVM = []
for i in range(0,5):
    X_train = trainFold[i][["XCoord","YCoord","DistFromCenter"]]
    y_train = trainFold[i][["Competitor"]]
    X_test = testFold[i][["XCoord","YCoord","DistFromCenter"]]
    y_test = testFold[i][["Competitor"]]
    clf_KNN = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
    clf_KNN.fit(X_train, y_train)
    y_pred_KNN.append(clf_KNN.predict(X_test))
####"""SVM_model""""
    clf_SVM = SVC(kernel = 'linear', C = 1e5) # just a big number 
    clf_SVM.fit(X_train,y_train)
    y_pred_SVM.append(clf_SVM.predict(X_test))
#%%
"""Model KNN & SVM full dataset"""
X = train_dataset[["XCoord","YCoord","DistFromCenter"]]
y = train_dataset[["Competitor"]]
X_test = test_dataset[["XCoord","YCoord","DistFromCenter"]]
y_test = test_dataset[["Competitor"]]
#%%%
clf_KNN = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf_KNN.fit(X, y)
y_pred_full_KNN = clf_KNN.predict(X_test)
acc_KNN = accuracy_score(y_test, y_pred_full_KNN)
#%%%
clf_SVM = SVC(kernel = 'linear', C = 1e5) # just a big number 
clf_SVM.fit(X_train,y_train)
y_pred_full_SVM = clf_SVM.predict(X_test)
acc_SVM = accuracy_score(y_test, y_pred_full_SVM)
#%%
"""BUILD DATATRAIN- DATATEST META"""
dataset_train_meta = train_dataset.copy()
dataset_train_meta["M1"] = np.concatenate(y_pred_KNN)
dataset_train_meta["M2"] = np.concatenate(y_pred_SVM)
dataset_test_meta = test_dataset.copy()
dataset_test_meta["M1"] = y_pred_full_KNN
dataset_test_meta["M2"] = y_pred_full_SVM
#%%
"""Model logistice-regression"""
X_train_meta = dataset_train_meta[["XCoord","YCoord","DistFromCenter","M1","M2"]]
y_train_meta = dataset_train_meta[["Competitor"]]
X_test_meta = dataset_test_meta[["XCoord","YCoord","DistFromCenter","M1","M2"]]
y_test_meta = dataset_test_meta[["Competitor"]]
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_meta, y_train_meta)
Z = logreg.predict(X_test_meta)
acc_logreg = accuracy_score(y_test_meta,Z)
#%%
"""cross validation & maingid search"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
k = np.arange(10)+1
parameters = {'n_neighbors': k}
knn = neighbors.KNeighborsClassifier()
clf = grid_search.GridSearchCV(knn,parameters,cv=5)
acc_cv = clf.fit(X_train.values, y_train.values.ravel())
#print (acc_cv)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
y_pred_KNN = clf.predict(X_test.values)
acc = accuracy_score(y_pred_KNN, y_test)
#%%
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([1, 2, 3, 4])
X_train = []
X_test = []
y_train = []
y_test = []
kf = KFold(len(X), n_folds=5)
len(kf)
print(kf)  
KFold(len(X), n_folds=5, shuffle=False,random_state=None)
for train_index, test_index in kf:
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train.append(X.ix[train_index,:])
   X_test.append(X.ix[test_index,:]) 
   y_train.append(y.ix[train_index,:])
   y_test.append(y.ix[test_index,:])      
#%%
#k = np.arange(10)+1
#
#parameters = {'n_neighbors': k}
avgMax = 0
kneed = -1
for j in range(1,20):
    knn = neighbors.KNeighborsClassifier(n_neighbors = j, p = 2)
    #clf = grid_search.GridSearchCV(knn,parameters,cv=5)
    #knn = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
    k_fold = KFold(5)
    result = []
    print(j)
    for i, (train, test) in enumerate(k_fold.split(X, y)):
        knn.fit(X.ix[train,:].values, y.ix[train,:].values.ravel())
        y_pred = knn.predict(X.ix[test,:].values)
        print(accuracy_score(y.ix[test,:], y_pred))
        result.append(accuracy_score(y.ix[test,:], y_pred))
    

    avg = np.average(result)
    if (avg > avgMax):
        avgMax = avg
        kneed = j
print(kneed)
print(avgMax)

  
        #result.append("[fold {0}]  {1} {2}".
          #      format(i,  clf.best_params_,accuracy_score(y.ix[test,:], y_pred)))
    #print("cho  accuracy_score(y.ix[test,:], y_pred)
    
    
    
    
    
#%%
