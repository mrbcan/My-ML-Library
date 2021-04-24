# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler=pd.read_csv('Churn_Modelling.csv')

X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values

#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le=preprocessing.LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])


le2=preprocessing.LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#birden fazla heterojen yapıdaki kolonu aynı anda dönüştürür

ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],
remainder='passthrough')

X=ohe.fit_transform(X)
X=X[:,1:]

#test and train (verileri bölme)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.33,random_state=0)


#XG BOOST ALGORİTMASI
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)


















