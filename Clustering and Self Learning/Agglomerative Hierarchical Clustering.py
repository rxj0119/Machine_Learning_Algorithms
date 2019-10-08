#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import pandas as pd
import math


# In[15]:


D = [ ((170, 57, 32),'W'),
((192, 95, 28), 'M'),
((150, 45, 30), 'W'),
((170, 65, 29),'M'),
((175, 78, 35),'M'),
((185, 90, 32), 'M'),
((170, 65, 28), 'W'),
((155, 48, 31), 'W'),
((160, 55, 30), 'W'),
((182, 80, 30), 'M'),
((175, 69, 28), 'W'),
((180, 80, 27), 'M'),
((160, 50, 31), 'W'),
((175, 72, 30), 'M'), ]

for x in D:
    print(x[0])

index_arr = []
for x in range(len(D)):
    index_arr.append(x)

# Getting height to a dataframe
height = []
for x in D:
    height.append(x[0][0])
    
# Getting weight to a dataframe
weight = []
for x in D:
    weight.append(x[0][1])
    
# Getting age to a dataframe
age = []
for x in D:
    age.append(x[0][2])

# Merging the data cols
h = pd.Series(height,index=index_arr)
w = pd.Series(weight,index=index_arr)
a = pd.Series(age,index=index_arr)
df1 = pd.concat([h, w,a ], axis=1) # Holds the input data
df1.columns = ['Height','Weight','Age']

# Calculating Eclidean Distance measure matrix (Entire Matrix)
from scipy.spatial.distance import pdist, squareform

distances = pdist(df1.values, metric='euclidean')
dist_matrix = squareform(distances)

# Building the dataframe
dataframe_ = {}
for x in range(len(D)):
    c_name = ''
    c_name += 'Col_'
    num = str(x)
    c_name += num

    dataframe_[c_name] = dist_matrix[:,x]
# Extracting the upper triangle only
dataset_matrix = pd.DataFrame(dataframe_)
df = dataset_matrix.where(np.tril(np.ones(dataset_matrix.shape)).astype(np.bool))
# Finding the minimum value in all rows
min_values_of_rows = (df.transpose().where(df.transpose().gt(0)).min(0))

print(df)


# In[175]:


# Finding the minimum value of the matrix
small = 10 
for x in min_values_of_rows:
    if x < small and math.isnan(x) is not True:
        small = x
print(small)


# In[176]:


# Merging the min P4,P7
print(df1.loc[[6]])
print(df1.loc[[3]])


# In[324]:


# Going over row 6
for x in range(len(df.columns)):
    if math.isnan(df.loc[6][x]) is not True:
        if (float)(df.loc[6][x])> (float)(df.loc[3][x]):
            df.loc[6][x] = df.loc[3][x]

# Going over row 3
for x in range(len(df.columns)):
    if math.isnan(df.loc[3][x]) is not True:
        if (float)(df.loc[3][x])> (float)(df.loc[6][x]):
            df.loc[3][x] = df.loc[6][x]


            
# Going over col 6
for x in range(len(df)):
    if math.isnan(df.loc[x][6]) != True:
        if (float)(df.loc[x][6])> (float)(df.loc[x][3]):
            df.loc[x][6] = df.loc[x][3]
            
# Going over col 3
for x in range(len(df.columns)):
    if math.isnan(df.loc[x][3]) is not True:
        if (float)(df.loc[x][3])> (float)(df.loc[x][6]):
            df.loc[x][3] = df.loc[x][6]            

print(df)


# In[325]:


# Merging the min P10,P12
print(df1.loc[[9]])
print(df1.loc[[11]])


# In[326]:


# Going over row 9
for x in range(len(df.columns)):
    if math.isnan(df.loc[9][x]) is not True:
        if (float)(df.loc[9][x])> (float)(df.loc[11][x]):
            df.loc[9][x] = df.loc[11][x]

# Going over row 11
for x in range(len(df.columns)):
    if math.isnan(df.loc[11][x]) is not True:
        if (float)(df.loc[11][x])> (float)(df.loc[9][x]):
            df.loc[11][x] = df.loc[9][x]


            
# Going over col  9
for x in range(len(df)):
    if math.isnan(df.loc[x][9]) != True:
        if (float)(df.loc[x][9])> (float)(df.loc[x][11]):
            df.loc[x][9] = df.loc[x][11]
            
# Going over col 11
for x in range(len(df.columns)):
    if math.isnan(df.loc[x][11]) is not True:
        if (float)(df.loc[x][11])> (float)(df.loc[x][9]):
            df.loc[x][11] = df.loc[x][9]   

print(df)


# In[327]:


# Merging the min P11,P14
print(df1.loc[[10]])
print(df1.loc[[13]])


# In[328]:


# Going over row 10
for x in range(len(df.columns)):
    if math.isnan(df.loc[10][x]) is not True:
        if (float)(df.loc[10][x])> (float)(df.loc[13][x]):
            df.loc[10][x] = df.loc[13][x]

# Going over row 13
for x in range(len(df.columns)):
    if math.isnan(df.loc[13][x]) is not True:
        if (float)(df.loc[13][x])> (float)(df.loc[10][x]):
            df.loc[13][x] = df.loc[10][x]


            
# Going over col  10
for x in range(len(df)):
    if math.isnan(df.loc[x][10]) != True:
        if (float)(df.loc[x][10])> (float)(df.loc[x][13]):
            df.loc[x][10] = df.loc[x][13]
            
# Going over col 13
for x in range(len(df.columns)):
    if math.isnan(df.loc[x][13]) is not True:
        if (float)(df.loc[x][13])> (float)(df.loc[x][10]):
            df.loc[x][13] = df.loc[x][10]   

print(df)


# In[329]:


# Merging the min P9,P13
print(df1.loc[[8]])
print(df1.loc[[12]])


# In[330]:


# Going over row  8 
for x in range(len(df.columns)):
    if math.isnan(df.loc[8][x]) is not True:
        if (float)(df.loc[8][x])> (float)(df.loc[12][x]):
            df.loc[8][x] = df.loc[12][x]

# Going over row  12
for x in range(len(df.columns)):
    if math.isnan(df.loc[12][x]) is not True:
        if (float)(df.loc[12][x])> (float)(df.loc[8][x]):
            df.loc[12][x] = df.loc[8][x]


            
# Going over col  8 
for x in range(len(df)):
    if math.isnan(df.loc[x][8]) != True:
        if (float)(df.loc[x][8])> (float)(df.loc[x][12]):
            df.loc[x][8] = df.loc[x][12]
            
# Going over col 12
for x in range(len(df.columns)):
    if math.isnan(df.loc[x][12]) is not True:
        if (float)(df.loc[x][12])> (float)(df.loc[x][8]):
            df.loc[x][12] = df.loc[x][8]   

print(df)


# In[331]:


# Merging the min P2,P6
print(df1.loc[[1]])
print(df1.loc[[5]])


# In[332]:


# Going over row  1
for x in range(len(df.columns)):
    if math.isnan(df.loc[1][x]) is not True:
        if (float)(df.loc[1][x])> (float)(df.loc[5][x]):
            df.loc[1][x] = df.loc[5][x]

# Going over row  5
for x in range(len(df.columns)):
    if math.isnan(df.loc[5][x]) is not True:
        if (float)(df.loc[5][x])> (float)(df.loc[1][x]):
            df.loc[5][x] = df.loc[1][x]


            
# Going over col   1
for x in range(len(df)):
    if math.isnan(df.loc[x][1]) != True:
        if (float)(df.loc[x][1])> (float)(df.loc[x][5]):
            df.loc[x][1] = df.loc[x][5]
            
# Going over col 5
for x in range(len(df.columns)):
    if math.isnan(df.loc[x][5]) is not True:
        if (float)(df.loc[x][5])> (float)(df.loc[x][1]):
            df.loc[x][5] = df.loc[x][1]   

print(df)

