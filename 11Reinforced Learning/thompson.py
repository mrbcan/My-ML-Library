# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:52:40 2020

@author: Mr Bcan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
veriler=pd.read_csv('Ads_CTR_Optimisation.csv')

import math
#UCB
N=10000 #10.000 tıklama
d=10 #toplam 10 ilan var
#Ri(n)
#Ni(n)

toplam=0 #toplam odul
secilenler=[]
birler=[0]*d
sifirlar=[0]*d

for n in range(1,N):
    ad=0 #secilen ilan
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1, sifirlar[i]+1)
        if rasbeta >max_th:
            max_th=rasbeta
            ad=i
    secilenler.append(ad)
    odul=veriler.values[n,ad] #verierdeki n.satır=1 ise odul 1
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1
        
    toplam=toplam+odul
    
print('Toplam odul:',toplam)

plt.hist(secilenler)
plt.show()