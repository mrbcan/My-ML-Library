# -*- coding: utf-8 -*-

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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

print('Linear r2 değeri')
print(r2_score(Y,lin_reg.predict(X)))


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

print('Polynomial r2 değeri')
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

#verilerin ölçeklendirme scaler
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

#svr da scaler kullanilmasi onemli
from sklearn.svm import SVR

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

print('svr r2 değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

#Decision tree reg.
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X+0.5
K=X-0.4

plt.scatter(X, Y,color='purple')
plt.plot(x,r_dt.predict(X),color='red')
#SONUCLAR MUHAKKAK O AĞAÇTAKİLER ÇICKAK
plt.scatter(x,r_dt.predict(Z) ,color='yellow')
plt.plot(x,r_dt.predict(K))
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

print('decision tree r2 değeri')
print(r2_score(Y,r_dt.predict(X)))

#Random Forest reg.
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=(0))

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')
plt.plot(x,rf_reg.predict(Z),color='green')

#R2 ile yöntemleri karşılaştırabiliyoruz









