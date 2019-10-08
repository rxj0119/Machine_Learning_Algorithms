#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np


# In[63]:


# Traing model with labled data 
Ds = { ((170, 57, 32), 'W'),
((190, 95, 28), 'M'),
((150, 45, 35), 'W'),
((168, 65, 29), 'M'),
((175, 78, 26), 'M'),
((185, 90, 32), 'M'),
((171, 65, 28), 'M'),
((155, 48, 31), 'M'),
((165, 60, 27), 'M') }

D = Ds
X = []
Y = []
Z = []
Dic = []
for x in D:
    if x[1] == 'M':
        X.append(x[0][0])
        Y.append(x[0][1])
        Z.append(x[0][2])
    else:
        X.append(-x[0][0])
        Y.append(-x[0][1])
        Z.append(-x[0][2])

for x in D:
    if x[1] == 'M':
        Dic.append(1)
    else:
        Dic.append(0)

# Scatter plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X, Y, Z, c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

xarray = []
yarray = Dic
for i in D:
    xarray.append([i[0][0],i[0][1],i[0][2]])

step_size_alpha = 0.0001
iterations = 100

class LogisticRegresion:
    def gradient_desent(self,x,y):
        w = np.ones((np.shape(x)[1],1))
        for i in range(iterations):
            w = w - step_size_alpha * x.transpose() * (self.sigmoid(x*w) -y)
        return w

    def classify(self,x,w):
        prob = self.sigmoid(sum(x*w))
        classification = 0
        if prob >0.5: classification = 1
        return [classification,prob.__str__()]
    
    def prob(self,x,w):
        prob = self.sigmoid(sum(x*w))
        prob = prob.__str__()     
        stra =''      
        for x in prob:
            if x != '[' and x != ']':
                stra +=x               
        prob = float(stra)
        if prob < 0.5:
            return 1-prob
        else:
            return prob
       
    def sigmoid(self,ws): return 1.0/(1+np.exp(-ws))

def Sort(sub_li): 
    l = len(sub_li) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (sub_li[j][1] > sub_li[j + 1][1]): 
                tempo = sub_li[j] 
                sub_li[j]= sub_li[j + 1] 
                sub_li[j + 1]= tempo 
    return sub_li    
    
logicReg = LogisticRegresion()
warray = logicReg.gradient_desent(np.mat(xarray),np.mat(yarray).transpose())

# Test the classifier

Dt = [ ((169, 58, 30), 'W'),
((185, 90, 29), 'M'),
((148, 40, 31), 'W'),
((177, 80, 29), 'M'),
((170, 62, 27), 'W'),
((172, 72, 30), 'M'),
((175, 68, 27), 'W'),
((178, 80, 29), 'M') ]


correct = 0
total = 0

for x in Dt:
    predict = logicReg.classify(x[0],warray)
    if predict[0] == 1 and x[1] == 'M':
        correct+=1
    if predict[0] == 0 and x[1] == 'W':
        correct+=1
    total+=1

print('Accuracy = ',correct/total * 100)



