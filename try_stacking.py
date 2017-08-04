import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
#from pyfm import pylibfm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import timeit
from scipy import stats
from sklearn.neural_network import MLPClassifier
import keras.utils.np_utils as np_utils
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold
from sklearn import neighbors, datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
#%%
""" Load data """
train_t4 = pd.read_csv('/home/lanhpth/WorkSpace/Internet/model/model_2017-05/dataset_Internet_load_attend_session_daily_201704.csv')
test_t5 =  pd.read_csv('/home/lanhpth/WorkSpace/Internet/model/model_2017-05/predictset_Internet_load_attend_session_daily_201705.csv')

#%%
""" Data Processing """
def disasembly_set(path, name, sel_col, ignore_cols=[] , class_col = None):
    df = pd.read_csv(path + name)
    df = df[sel_col]
    df = df.select(lambda x: x not in ignore_cols, axis=1)
    df = df.fillna(-1)
    
    dfX = df.select(lambda x: x != class_col, axis=1)
#    df_num = dfX.select_dtypes(exclude=[np.number]).columns.tolist()
#    for k in df_num:
#        dfX[k] = dfX[k].factorize()[0]
        
    X = dfX.as_matrix().astype(np.float32)
    if class_col is not None:
        le = preprocessing.LabelEncoder()
        Y = pd.Series(le.fit_transform(df[class_col]))
        y = np_utils.to_categorical(Y)
    else:
        Y = None
        y = None
    
    return (X, Y, y)

dataset = {'path': '/home/lanhpth/WorkSpace/Internet/model/model_2017-05/',
           'train': 'dataset_Internet_load_attend_session_daily_201704.csv',
           'test': 'predictset_Internet_load_attend_session_daily_201705.csv',
           'id_col': 'Contract',
           'class_col': 'ContractStatus'}

sel_col = ['ContractStatus', "DownloadLim", "UploadLim","ssOnline_Max","ssOnline_Min","ssOnline_Std","Session_Count",
           "Size27Upload","Size27Download","ssOnline_Mean","Diff2Download","Diff12Download",
           "Diff3Download","Size0Download","Diff1Download","Diff1Upload","Diff27Upload",
           "Diff7Download","Diff5Download","Diff4Download","Diff27Download","Diff19Download"]  
#sel_col = [ "DownloadLim", "UploadLim","ssOnline_Max","ssOnline_Min","ssOnline_Std","Session_Count",
#           "Size27Upload","Size27Download","ssOnline_Mean","Diff2Download","Diff12Download",
#           "Diff3Download","Size0Download","Diff1Download","Diff1Upload","Diff27Upload",
#           "Diff7Download","Diff5Download","Diff4Download","Diff27Download","Diff19Download"]          
Xtrain, Ytrain, _ = disasembly_set(path=dataset['path'], name=dataset['train'], class_col=dataset['class_col'], sel_col = sel_col, ignore_cols=[dataset['id_col']])
#Xtest, Ytest, _ = disasembly_set(path=dataset['path'], name=dataset['test'], sel_col = sel_col, ignore_cols=[dataset['id_col']])
#%%
""" OverSampling """
sm = SMOTE(1.0)
X_res, y_res = sm.fit_sample(Xtrain, Ytrain)
#%%
"""Find k in knn"""
avgMax = 0
kneed = -1
for j in range(1,10):
    knn = neighbors.KNeighborsClassifier(n_neighbors = j, p = 2)
    #clf = grid_search.GridSearchCV(knn,parameters,cv=5)
    #knn = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
    k_fold = KFold(5)
    result = []
    print(j)
    for i, (train, test) in enumerate(k_fold.split(Xtrain, Ytrain)):
        knn.fit(Xtrain[train], Ytrain[train])
        y_pred = knn.predict(Xtrain[test])
        print(accuracy_score(Ytrain[test], y_pred))
        result.append(accuracy_score(Ytrain.ix[test,:], y_pred))
    avg = np.average(result)
    if (avg > avgMax):
        avgMax = avg
        kneed = j
print(kneed)
print(avgMax)
#%%
""" try KNN with k =1"""
k_fold = KFold(5)
precision_knn = []   
recall_knn = []
Pred_knn = []
precision_svm = []
recall_svm = []
Pred_svm = []
knn = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf_SVM = SVC(kernel = 'linear', C = 1e5) # just a big number 
for k, (train, test) in enumerate(k_fold.split(Xtrain, Ytrain)):
    knn.fit(Xtrain[train], Ytrain[train])
    y_pred = knn.predict(Xtrain[test])
    precision_knn.append(precision_score(Ytrain[test], y_pred))
    recall_knn.append(recall_score(Ytrain[test], y_pred))
    Pred_knn.append(y_pred)    
    clf_SVM.fit(Xtrain[train], Ytrain[train]) #try SVM
    y_pred_svm = clf_SVM.predict(Xtrain[test])
    precision_knn.append(precision_score(Ytrain[test], y_pred_svm))
    recall_knn.append(recall_score(Ytrain[test], y_pred_svm))
    Pred_svm.append(y_pred_svm)
#%%
""" XGBoost """
k_fold = KFold(5)
precision_xgbg = []   
recall_xgb = []
pred_xgb = []
xgbm = xgb.XGBClassifier(nthread= 32, learning_rate= 0.01, n_estimators= 100, max_depth= 120, min_child_weight= 1, 
                         gamma= 0.1, subsample= 0.8, colsample_bytree= 0.3, scale_pos_weight= 1, reg_alpha= 1e-5)
for k, (train, test) in enumerate(k_fold.split(X_res, y_res)):    
    xgbm.fit(X_res[train], y_res[train])
    y_pred1 = xgbm.predict(X_res[test])
    precision_xgbg.append(precision_score(y_res[test], y_pred1))
    recall_xgb.append(recall_score(y_res[test], y_pred1))
    pred_xgb.append(y_pred1)
#print(xgbm.feature_importances_)

#%%
"""AdaBoost"""
k_fold = KFold(5)
precision_adaboost = [] 
recall_adaboost = []
pred_adaboost = []
bdt = AdaBoostClassifier(learning_rate= 0.01, n_estimators= 100)
for k, (train, test) in enumerate(k_fold.split(X_res, y_res)):    
    bdt.fit(X_res[train], y_res[train])
    y_pred2 = xgbm.predict(X_res[test])
    precision_adaboost.append(precision_score(y_res[test], y_pred2))
    recall_adaboost.append(recall_score(y_res[test], y_pred2))
    pred_adaboost.append(y_pred2)
#%%























