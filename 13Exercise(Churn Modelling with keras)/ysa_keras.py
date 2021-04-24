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


#verilerin ölçeklendirme 0ve 1 aralığına sıkıştırmak için
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#YAPAY SİNİR AĞI -YSA  (KERAS)

import keras
from keras.models import Sequential
from keras.layers import Dense #katman

classifier=Sequential()
#giriş katmanı oldu için input_dim belirtcez
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim=11))    
#yeni gizli bir katman
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))    #(11+1)x+y/2 genelde :D ,init ilk değeri verir 0a yakın,
#çıkış katmanı
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid')) 
#nöron ve katmanlarının synapsizlerini nasıl optimize edileceği hangi fonk,metric
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'],)

classifier.fit(X_train,y_train,epochs=50)#epoc hs çalışırma miktarı

y_pred=classifier.predict(X_test)
#bırakır mı bırakmaz mı 1veya 0 olarak bölücez
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)


















