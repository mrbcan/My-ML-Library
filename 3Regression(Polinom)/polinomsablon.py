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

#linear regression-doğrusal model olusturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


#polynomial regression-doğrusal olmayan model olusturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


#görselleştirmeler sırasıyla(linear ve 4.dereceden)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()


plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(x_poly),color='blue')
plt.show()

#tahminler

print(lin_reg.predict([[6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6]])))