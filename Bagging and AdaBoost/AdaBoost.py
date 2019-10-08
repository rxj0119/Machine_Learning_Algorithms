#!/usr/bin/env python
# coding: utf-8

# In[1]:

'''Shreyash Shrivastava
   1001397477'''

import numpy as np
import pandas as pd
import os
import math


# In[125]:


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

# Assigining probabilities
# The function outputs a sample after each iteration on Adaboost, with the assigned probabilities to each data instance

def adaBoost_data(data,probability):

    choosing_from_data = []
    
    for x in range(len(data)):
        choosing_from_data.append(x)
    
    boosting_data_indices = np.random.choice( choosing_from_data, len(data), probability)
    
    boosting_data_array = []
    
    for x in boosting_data_indices:
        boosting_data_array.append(data.iloc[x])
        
    
 
    return boosting_data_array[:100]

def adaBoost_test_data(data):

    choosing_from_data = []
    
    for x in range(len(data)):
        choosing_from_data.append(x)
    
    #boosting_data_indices = np.random.choice( choosing_from_data, len(data), probability)
    
    boosting_data_array = []
    
    for x in choosing_from_data:
        #boosting_data_array.append(data.iloc[x])
        boosting_data_array.append(data.iloc[x])
    
 
    return boosting_data_array[:100]

##
# Loading the testing text file in the pandas dataframe
DATA_PATH = os.path.join(os.getcwd(),'DATA')
def load_data_testingn(txt_path=DATA_PATH):
    txt_path = os.path.join(txt_path,'test.txt')
    return pd.read_csv('test.txt', sep=",", header=None) 
testing_data_ada = load_data_testingn()
testing_data_ada.columns = ['Variance','Skew','Curtosis','Entropy','Label']


##
# Assigning the probabilities array ( 1/len(data) in the first iteration)
probability = []
for x in range(len(data)):
    probability.append(1/len(data))

data_np = []
data_train = adaBoost_data(data,probability)
for x in data_train:
    data_np.append(x)
data_np = np.array(data_np)

# Training the logistic regression classifier

arr =[]
test_arr = []
prediction_array_of_models = []

alpha_array = [] # for each model 

# T is the number of iterations 
# Tested with (1,10,50,100) 
for t in range(100):
    # Attribute values 
    xarray = []

    # Labels
    yarray = []


    for x in range(len(data_np)):
        xarray.append([data_np[x][0]])

    
    count = 0  
    for x in range(len(data_np)):
        xarray[count].append(data_np[x][1])
        count+=1
        

    count = 0
    for x in range(len(data_np)):
        xarray[count].append(data_np[x][2])
        count+=1

    count = 0
    for x in range(len(data_np)):
        xarray[count].append(data_np[x][3])
        count+=1

    count =0
    for x in range(len(data_np)):
        yarray.append(data_np[x][4])
        count +=1

    arr =xarray
    
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
    
    # Testing the Trained model with the training data

    data_test = adaBoost_test_data(data)
    xarray_test = []
    # Testing Labels
    yarray_test = []
    
    for x in range(len(data_np)):
        xarray_test.append([data_np[x][0]])

    
    count = 0  
    for x in range(len(data_np)):
        xarray_test[count].append(data_np[x][1])
        count+=1
        

    count = 0
    for x in range(len(data_np)):
        xarray_test[count].append(data_np[x][2])
        count+=1

    count = 0
    for x in range(len(data_np)):
        xarray_test[count].append(data_np[x][3])
        count+=1

    count =0
    for x in range(len(data_np)):
        yarray_test.append(data_np[x][4])
        count +=1
        
    predictions = []
    
    test_arr = xarray_test
    for x in xarray_test:
        predictions.append(logicReg.classify(x,warray))
        
    # Accuracy for model
    co = 0
    true_pos = 0
    misclassified_indices = []
    while (co<len(yarray_test)):
        if (predictions[co] == yarray_test[co]):
            true_pos +=1
        else:
            misclassified_indices.append(co)
        co+=1
    
    # Computing epsilon, error of the hypothesis 
    # E[t] is the summation over the training examples 
    
    # e = d(ti) -> probability of the ith training example in the sample * delta(i) -> misclassified 1 otherwise 0
    epsilon_ = 0
    
    for x in range(len(data_test)):
        if x in misclassified_indices:
            epsilon_ += probability[x] *1
        else:
            epsilon_ += probability[x] * 0
        
     # Calulating alpha 
    aplha_ = 1/2 * math.log( (1-epsilon_) / epsilon_)
    alpha_array.append(aplha_)
    #print(aplha_)
    
    # Computing the new probability distribution
    new_prob_dist = []
    co = 0
    for x in range(len(data_test)):
        agree = 0
        if predictions[x] == yarray_test[x]:
            co +=1
            agree = 1
            new_prob_dist.append( probability[x] * math.exp(-1 * aplha_ * agree) )
        else:
            agree = -1
            new_prob_dist.append( probability[x] * math.exp(-1 * aplha_ * agree) )
    
    # Normalization factor Z(t) = sum( dt * yt * ht )
    zed_t = sum(new_prob_dist)
   
    
   
    for x in range(len(new_prob_dist)):
        new_prob_dist[x] = new_prob_dist[x] / zed_t
    
    # Prediction data for each iteration
    def prediction(data_):
        
        prediction = []
        xarray_test = []
        # Testing Labels
        yarray_test = []

        for x in data_['Variance']:
            xarray_test.append([x])

        count = 0
        for x in data_['Skew']:
            xarray_test[count].append(x)
            count+=1

        count = 0
        for x in data_['Curtosis']:
            xarray_test[count].append(x)
            count+=1

        count = 0
        for x in data_['Entropy']:
            xarray_test[count].append(x)
            count+=1

        for x in data_['Label']:
            yarray_test.append(x)


        for x in xarray_test:
            predictions.append(logicReg.classify(x,warray))
        
        return predictions
    prediction_array_of_models.append(prediction(testing_data_ada))
    
    for x in range(len(prediction_array_of_models[t])):
        if prediction_array_of_models[t][x] == 0:
            prediction_array_of_models[t][x] = -1
        
     
    probability = new_prob_dist

   

# Final classifier
final_out_put = []
co = 0

for x in range(len(prediction_array_of_models)):
    prediction_array_of_models[x] = prediction_array_of_models[x][100:160]

print(len(prediction_array_of_models[0]))

for i in range(len(prediction_array_of_models)):
    for y in range(len(prediction_array_of_models[i])):
        prediction_array_of_models[i][y] *= alpha_array[i] 
    
for y in range(len(prediction_array_of_models[0])):
    res_pos = 0
    res_neg = 0
    for x in range(len(prediction_array_of_models)):
        if prediction_array_of_models[x][y]>0:
            res_pos += prediction_array_of_models[x][y]
        else:
            res_neg += abs(prediction_array_of_models[x][y])
    
    if res_pos > res_neg:
        final_out_put.append(1)
    else:
        final_out_put.append(0)

def load_data_testingn(txt_path=DATA_PATH):
    txt_path = os.path.join(txt_path,'test.txt')
    return pd.read_csv('test.txt', sep=",", header=None) 
testing_data_ada = load_data_testingn()
testing_data_ada.columns = ['Variance','Skew','Curtosis','Entropy','Label']

# Final accuracy 

counting = 0


for x in range(len(testing_data_ada['Label'])):
    if testing_data_ada['Label'][x] == final_out_put[x]:
        counting+=1
print('Accuracy: ' , counting/ len(testing_data_ada['Label']))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




