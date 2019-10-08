#!/usr/bin/env python
# coding: utf-8

# In[2]:
'''Shreyash Shrivastava
   1001397477'''


import numpy as np
import pandas as pd
import os


# In[3]:


# Loading the training text file in the pandas dataframe
DATA_PATH = os.path.join(os.getcwd(),'DATA')
def load_data(txt_path=DATA_PATH):
    txt_path = os.path.join(txt_path,'train.txt')
    return pd.read_csv('train.txt', sep=",", header=None) 
data = load_data()
data.columns = ['Variance','Skew','Curtosis','Entropy','Label']

# Loading the testing text file in the pandas dataframe
DATA_PATH = os.path.join(os.getcwd(),'DATA')
def load_data(txt_path=DATA_PATH):
    txt_path = os.path.join(txt_path,'test.txt')
    return pd.read_csv('test.txt', sep=",", header=None) 
data_test = load_data()
data_test.columns = ['Variance','Skew','Curtosis','Entropy','Label']


# In[308]:


# Spliting the data into baggs for bootstrap aggregation
def split_train_set(data,bags):
    
    bagging_set = []
    
    bag_size = int(len(data) * (bags/100))
    
    for x in range(bags):
        shuffled_indices = np.random.permutation(len(data))
       
        data_indices = (int)(0.01 * len(data))
        bagging_set.append(data.iloc[shuffled_indices[:(int)(0.2 * len(data))]])
    
    return bagging_set

bag_of_data = (split_train_set(data,1))


# Implementing logistic regression classifier each data set in the bag

bagging_predictions = []

for x in range(len(bag_of_data)):
    bagging_predictions.append([])

for t in range(len(bag_of_data)):
    # Attribute values 
    xarray = []

    # Labels
    yarray = []


    for x in bag_of_data[t]['Variance']:
        xarray.append([x])

    count = 0
    for x in bag_of_data[t]['Skew']:
        xarray[count].append(x)
        count+=1

    count = 0
    for x in bag_of_data[t]['Curtosis']:
        xarray[count].append(x)
        count+=1

    count = 0
    for x in bag_of_data[t]['Entropy']:
        xarray[count].append(x)
        count+=1

    for x in bag_of_data[t]['Label']:
        yarray.append(x)



    # Hyper-parameters
    step_size_alpha = 0.9
    iterations = 10

    # Logistic Classifier
    class LogisticRegresion:
        def gradient_desent(self,x,y):
            w = np.ones((np.shape(x)[1],1))
            for i in range(iterations):
                w = w - step_size_alpha * x.transpose() * (self.sigmoid(x*w) -y)

            return w

        def classify(self,x,w):
            prob = self.sigmoid(sum(x*w))
            #classification = 'Probability: '+ prob.__str__() + '\nClassified as Not Authentic\n'
            #if prob >0.5: classification='Probability:'  + prob.__str__() + '\nClassified as Authentic\n'
            classification = 0
            if prob >0.5: classification = 1
            return classification

        def sigmoid(self,ws): return 1.0/(1+np.exp(-ws))

    logicReg = LogisticRegresion()
    warray = logicReg.gradient_desent(np.mat(xarray),np.mat(yarray).transpose())

    # Testing the Trained model with the data and adding the predictions into an array (to implement maximum voting)

    xarray_test = []
    # Testing Labels
    yarray_test = []

    for x in data_test['Variance']:
        xarray_test.append([x])
   
    count = 0
    for x in data_test['Skew']:
        xarray_test[count].append(x)
        count+=1

    count = 0
    for x in data_test['Curtosis']:
        xarray_test[count].append(x)
        count+=1

    count = 0
    for x in data_test['Entropy']:
        xarray_test[count].append(x)
        count+=1

    for x in data_test['Label']:
        yarray_test.append(x)

    
    for x in xarray_test:
        bagging_predictions[t].append(logicReg.classify(x,warray))

    # Accuracy for each model for debugging
    co = 0
    true_pos = 0
    while (co<len(yarray_test)):
        if (bagging_predictions[t][co] == yarray_test[co]):
            true_pos +=1
        co+=1
    
    #print(true_pos/len(yarray_test))
    


    
bagging_predictions = np.array(bagging_predictions)


# Voting 

resulting_prediction = []
for x in range(len(bagging_predictions[0])):
    resulting_prediction.append(np.bincount(bagging_predictions[:,x]).argmax())

print(bagging_predictions)    

#print(resulting_prediction)
print(np.bincount(bagging_predictions[:,0]).argmax())
accuracy_ = []

for x in resulting_prediction:
    accuracy_.append([x])

c = 0
for x in yarray_test:
    accuracy_[c].append(x)
    c+=1


true_count = 0
total_count = len(accuracy_)
for x in accuracy_:
    if x[0] == x[1]:
        true_count+=1
        
print('True Positives count :',true_count)
print('Total Data count :',total_count)
print('Accuracy ', true_count/total_count)

