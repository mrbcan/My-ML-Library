# -*- coding: utf-8 -*-

#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri kümesi
veriler=pd.read_csv('Wine.csv')
X=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13].values


#test and train (verileri bölme)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#verilerin ölçeklendirme 0ve 1 aralığına sıkıştırmak için
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2) #2 boyuta (colona ) indirgendi

X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
#pca dönüşümünden önce gelen gelen LR
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen gelen LR
classifier_pca=LogisticRegression(random_state=0)
classifier_pca.fit(X_train2,y_train)

#tahminler
y_pred=classifier.predict(X_test)
y_pred2=classifier_pca.predict(X_test2)

#confusion matrixle karşılaştırma PCA öncesi ve sonrası
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("PCA ' SIZ ")
print(cm)

cm2=confusion_matrix(y_test,y_pred2)
print('PCA')
print(cm2)

cm3=confusion_matrix(y_pred,y_pred2)
print('PCAsız ve PCA lı')
print(cm3)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)

X_train_lda=lda.fit_transform(X_train, y_train)#sınıfları öğrenmesiiçin yyi de ver
X_test_lda=lda.transform(X_test)

#lda dönüşümünden sonra gelen gelen LR
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA verisini tahmin et
y_pred_lda=classifier_lda.predict(X_test_lda)
#Lda sonrası / orijinal veri

print('LDA vs orjinal')
cm3=confusion_matrix(y_pred,y_pred_lda)
print(cm3)







