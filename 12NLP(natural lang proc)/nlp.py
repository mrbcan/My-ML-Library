# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# 1-preprocessing işlemleri veri önişlemee(stopwords,case,parsers(html))
yorumlar=pd.read_csv('Restaurant_Reviews2.csv',sep=r'\s*,\s*')
"""
np.isnan(yorumlar)
np.where(np.isnan(yorumlar))
np.nan_to_num(yorumlar)
"""


import re #regular expression ile kelimeleri temizleme
import nltk
from nltk.stem.porter import PorterStemmer#kelimeleri kökünü alır

ps=PorterStemmer()
nltk.download('stopwords')

from nltk.corpus import stopwords

derlem=[]
for i in range(716):
    yorum=re.sub('[^a-zA-Z]',' ',yorumlar['Column1'][i])
    yorum=yorum.lower()
    yorum=yorum.split() #♠kelimeleri böl listeye koy str->list
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum=' '.join(yorum)  #boşluk bırakarak stringe çevirdik tekrardan
    derlem.append(yorum)
    
# 2-Feature Extraction-->öznitelik çıkarımı(kelime sayıları,kelime grupları,n-gram,tf-ıdf)
#Bag of Words (BOW) 1ler ve 0dan kelimelerle çanta oluşturma
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=2000) #max 2bin kelime al baba,ramin sikilmesin
X=cv.fit_transform(derlem).toarray() #bağımsız değişken
y=yorumlar.iloc[:,1].values  #bağımlı değişken
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(y)
y=imputer.transform(y)
"""
# 3-Makine öğrenmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm) #%72.5 accuracy














