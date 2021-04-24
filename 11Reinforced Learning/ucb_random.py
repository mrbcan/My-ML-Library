# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:26:11 2020

@author: Mr Bcan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('Ads_CTR_Optimisation.csv')

"""
import random

N=10000
d=10
toplam=0
secilenler=[]

for n in range(0,N):
    ad=random.randrange(d) 
    secilenler.append(ad)
    odul=veriler.values[n,ad]#verierdeki n.satır=1 ise odul 1
    toplam=toplam+odul
    
plt.hist(secilenler)
plt.show()

"""

import math
#UCB
N=10000 #10.000 tıklama
d=10 #toplam 10 ilan var
#Ri(n)
oduller=[0]*d  #ilk basta butun ilanların odulu 0
#Ni(n)
tiklamalar=[0]*d #o ana kadarki tıklamalar
toplam=0 #toplam odul
secilenler=[]

for n in range(1,N):
    ad=0 #secilen ilan
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i] / tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb<ucb: #maxtan büyük bir ucb çıktı
           max_ucb=ucb
           ad=i
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad] #verierdeki n.satır=1 ise odul 1
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
    
print('Toplam odul:',toplam)

plt.hist(secilenler)
plt.show()

    
















