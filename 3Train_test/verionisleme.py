#kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri önişleme
#2.1 veri yükleme
veriler=pd.read_csv('veriler.csv')


#eksik veriler sci-kit learn
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veriler.iloc[:,1:4].values
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

#encoder: Nominal ordinal->Numeric
ulke=veriler.iloc[:,0:1].values


from sklearn import preprocessing

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])



ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

#numpy dizileri dataframe dönüşümü

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy",'kilo','yas'])


cinsiyet=veriler.iloc[:,-1].values
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
#dataframe birleştirme
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc2],axis=1)

#test and train (verileri bölme)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=.33,random_state=0)
print(x_train)

#verilerin ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)