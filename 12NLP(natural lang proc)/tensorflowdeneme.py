# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:19:01 2020

@author: Mr Bcan
"""


#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hello=tf.constant('HEllo , TEnsor')
sess=tf.Session()
print(sess.run(hello))



"""

def dels(first,second):
    a=len(first)
    b=len(second)
    
    if a >b :
        del first[:b]
        print(first)
        
    if a<b:
        c=[]
        c.append(second[0])
        c=c+first
        c=c+second[1:]
        print(c)
        
    if a==b:
        c=[]
        print(c)
        
        
dels([3,4,2,5,6,7],[1,3,4])    
dels([2,5],[3,4,7,8,9])        
dels([2,3,4],[8,7,9])







def dels(a,b):

    if(len(a)>len(b)):
        return a[len(b):]

    elif(len(b)>len(a)):
        return (b[0:1]+a[:]+b[1:])

    else:
        return []

print(dels([3,4,2,5,6,7],[1,3,4]))
print(dels([2,5],[3,4,7,8,9]))
print(dels([2,3,4],[8,7,9]))




def digits(num):

    if(num<10):
        return [num]

    else:
        last = digits(int(num%10))
        return (digits(int(num/10)))+last

print(digits(532))

def multiply(num,zeros):

    return digits(num)+digits(zeros)[1:]

print(multiply(5,1000))


print(digits(3))
    """


                                                                          