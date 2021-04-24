# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1.veri yukleme
veriler = pd.read_csv('odev_tenis.csv')
print(veriler)
#2.veri onisleme
#Toplu encoder Kategorik-> Numeric
from sklearn import preprocessing
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]
from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu=pd.DataFrame(data=c,index=range(14),columns=['overcast','rainny','sunny'])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)


#3-verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)


#Backward elimination
 
import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

#ilk (windy) at 
sonveriler=sonveriler.iloc[:,1:]
import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

X_l=sonveriler.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

#yeni predict yapma windy yi atma x_test ve train den 

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
#tekrar sistemi eğitme
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)














