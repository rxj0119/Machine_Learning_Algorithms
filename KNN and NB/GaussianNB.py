'''
    Shreyash Shrivastava
    1001397477
    CSE 6363

'''

import math

# Height Weight Age Class
D = [ ((170, 57, 32),'W'),
((192, 95, 28),'M'),
((150, 45, 30), 'W'),
((170, 65, 29), 'M'),
((175, 78, 35), 'M'),
((185, 90, 32), 'M'),
((170, 65, 28), 'W'),
((155, 48, 31), 'W'),
((160, 55, 30), 'W'),
((182, 80, 30), 'M'),
((175, 69, 28), 'W'),
((180, 80, 27), 'M'),
((160, 50, 31), 'W'),
((175, 72, 30), 'M'), ]

# Taking input from user
data_point = input('Enter the data point, separeted with a comma: Height, Weight\n')
data_point = list(data_point)

X =[]
for x in data_point:
    X.append(x)




# Probability of class W
no_of_w = 0
for x in D:
    if x[1] == 'W':
        no_of_w +=1

prob_of_w = (float(no_of_w)/float(len(D)))


# Probability of class M
no_of_m = 0
for x in D:
    if x[1] == 'M':
        no_of_m +=1

prob_of_m = (float(no_of_w)/float(len(D)))

# Creating a classifier from the training data
# male -> mean(height), sigma(height) ..
# female -> mean(height), sigma(height) ..

# @mean takes three arguments
# arg1 = Data set
# arg2 = gender as M or W
# arg3 = classifier as height, weight or age

def mean(D,gender,classifier):

    fetch = 0
    if classifier == 'height':
        fetch = 0
    if classifier == 'weight':
        fetch = 1
    if classifier == 'age':
        fetch = 2

    total_sum = 0
    data_sum = 0

    # data sum
    for x in D:
        if x[1] == gender:
            data_sum += x[0][fetch]

    # number of data points
    dp= 0
    for x in D:
        if x[1] == gender:
            dp +=1

    # total sum
    for x in D:
        total_sum += x[0][fetch]

    return float(float(data_sum)/float(dp))

# @variance takes three arguments
# arg1 = Data set
# arg2 = gender as M or W
# arg3 = classifier as height, weight or age


def variance(D,gender,classifier):
    fetch = 0
    if classifier == 'height':
        fetch = 0
    if classifier == 'weight':
        fetch = 1
    if classifier == 'age':
        fetch = 2
    mu = float(mean(D,gender,classifier))

    var = 0
    var = float(var)

    for x in D:
        if x[1] == gender:
            var += math.pow((float(x[0][fetch] - mu)),2)

    dp= 0

    for x in D:
        if x[1] == gender:
            dp +=1

    #return var
    var = float(float(var)/float(dp))

    return float(math.sqrt(var))


# male variable holds the gaussian distribution assumption for male and calculates mean and variance
male = [mean(D,'M','height'),variance(D,'M','height'),mean(D,'M','weight'),variance(D,'M','weight'),
        mean(D,'M','age'),
        variance(D,'M','age'),]



# female variable holds the gaussian distribution assumption for male and calculates mean and variance
female = [mean(D,'W','height'),variance(D,'W','height'),mean(D,'W','weight'),variance(D,'W','weight'),
          mean(D,'W','age'),
          variance(D,'W','age'),]

def calculateProbability(x, mean, stdev):
    exponent = float(math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))))
    return (1 / (float(math.sqrt(2*math.pi) * stdev))) * exponent

#  p(height|male)* p(wight|male) * p(age|male) from calculateProbability

print('P(height|M)', calculateProbability(X[0],male[0],male[1]))
print('P(Weight|M)', calculateProbability(X[1],male[2],male[3]))
print('P(Age|M)', calculateProbability(X[2],male[4],male[5]))
print('\n')

print('P(height|W)', calculateProbability(X[0],female[0],female[1]))
print('P(Weight|W)', calculateProbability(X[1],female[2],female[3]))
print('P(Age|W)', calculateProbability(X[2],female[4],female[5]))



p_m_x = calculateProbability(X[0],male[0],male[1]) * calculateProbability(X[1],male[2],male[3]) * calculateProbability(X[2],male[4],male[5]) * prob_of_m
p_w_x = calculateProbability(X[0],female[0],female[1]) * calculateProbability(X[1],female[2],female[3]) * calculateProbability(X[2],female[4],female[5]) * prob_of_w

print('\n')
print('P(M|X) ',p_m_x)
print('P(W|X) ',p_w_x)
print('\n')

if p_m_x > p_w_x:
    print('Predicted as : Male')
else:
    print('Predicted as : Female')