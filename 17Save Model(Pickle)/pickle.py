# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy
import pandas as pd
import matplotlib.pyplot as plt

#veri yükleme
veriler=pd.read_csv("maaslar.csv")
#data frame dilimleme(slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#Numpy array-dizi dönüşümü
X=x.values
Y=y.values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)



import pickle

dosya="model.kayit"

pickle.dump(lr,open(dosya,'wb'))


yuklenen=pickle.load(open(dosya,'rb'))
print(yuklenen.predict(X_test))                        