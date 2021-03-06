#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import random
# In[375]:
# Traing model with labled data 
Ds = { ((170, 57, 32), 'W'),
((190, 95, 28), 'M'),
((150, 45, 35), 'W'),
((168, 65, 29), 'M'),
((175, 78, 26), 'M'),
((185, 90, 32), 'M'),
((171, 65, 28), 'W'),
((155, 48, 31), 'W'),
((165, 60, 27), 'W') }

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
#print('Data Set: '+ '[155,40,35] ' +logicReg.classify([150,50,15],warray))

#'''Classify data in U with the trained classifier'''
U_predictions = []
Du = [ (182, 80, 30), (175, 69, 28), (178, 80, 27),
(160, 50, 31), (170, 72, 30), (152, 45, 29),
(177, 79, 28), (171, 62, 27), (185, 90, 30),
(181, 83, 28), (168, 59, 24), (158, 45, 28),
(178, 82, 28), (165, 55, 30), (162, 58, 28),
(180, 80, 29), (173, 75, 28), (172, 65, 27),
(160, 51, 29), (178, 77, 28), (182, 84, 27),
(175, 67, 28), (163, 50, 27), (177, 80, 30),
(170, 65, 28) ]


for x in Du:
    U_predictions.append(logicReg.classify(x,warray)[0])

# Confidence -> Find probabilities 
confid_prob = []
c = 0
for x in Du:
    confid_prob.append([c,logicReg.prob(x,warray)])
    c+=1
Sort(confid_prob)
confid_prob = confid_prob[::-1] 
#print(confid_prob)
# Now choosing U' from U (with labeles from the originl classifier) (Pool P) top 3 classified samples
U_dash = []

c = 0
while len(U_dash) <= 2: 
        U_dash.append([Du[confid_prob[c][0]],U_predictions[confid_prob[c][0]]])
        c+=1
        


# In[378]:


# Loop for I iterations 10 

Dl = [] # Labeled data

for x in Ds:
    if x[1] == 'M':
        Dl.append([x[0],1])
    if x[1] == 'W':
        Dl.append([x[0],0])

# Adding U' to labled data U' + L = L

for x in U_dash:
    Dl.append(x)

#Removing U' indices from U (unlabled data) U - U' = U
for x in U_dash:
    idx = 0
    for y in Du:
        if y == x[0]:
            del Du[idx]
        idx+=1

        
# Training the classifier on the labled data
# Repeating until Dl is empty

while len(Du)>0:
    # Train the classifier on newly made labled data Dl
    D = Dl
    X = []
    Y = []
    Z = []
    Dic = []
    for x in D:
        if x[1] == 1:
            X.append(x[0][0])
            Y.append(x[0][1])
            Z.append(x[0][2])
        else:
            X.append(-x[0][0])
            Y.append(-x[0][1])
            Z.append(-x[0][2])

    for x in D:
        if x[1] == 0:
            Dic.append(1)
        else:
            Dic.append(0)

    step_size_alpha = 0.0001
    iterations = 100
    
    logicReg = LogisticRegresion()
    #warray = logicReg.gradient_desent(np.mat(xarray),np.mat(yarray).transpose())
    
  
    U_predictions = []
    for x in Du:
        U_predictions.append(logicReg.classify(x,warray)[0])
    
    # Confidence -> Find probabilities 
    confid_prob = []
    c = 0
    for x in Du:
        confid_prob.append([c,logicReg.prob(x,warray)])
        c+=1
    Sort(confid_prob)
    confid_prob = confid_prob[::-1] 
    
    # Now choosing U' from U (with labeles from the originl classifier) (Pool P) top 3 classified samples
    U_dash = []
    c = 0
    if len(Du) >= 2:
        while len(U_dash) <= 2: 
                U_dash.append([Du[confid_prob[c][0]],U_predictions[confid_prob[c][0]]])
                c+=1
  
    if len(Du) == 1:
        c = 0
        U_dash.append([Du[confid_prob[c][0]],U_predictions[confid_prob[c][0]]])
        
    
    for x in U_dash:
        Dl.append(x)

    #Removing U' indices from U (unlabled data) U - U' = U
    for x in U_dash:
        idx = 0
        for y in Du:
            if y == x[0]:
                del Du[idx]
            idx+=1
    
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

