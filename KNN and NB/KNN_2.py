'''
    Shreyash Shrivastava
    1001397477
    CSE 6363

'''


import math


def euclideanDistance(item1, item2, length):
    cal_dist = 0
    for x in range(length):
        cal_dist += pow((item1[x] - item2[x]), 2)
    return math.sqrt(cal_dist)


def checkNeighbors(trainingData, test, k):
    distance_measure =[]
    length = len(test)
    for x in range(len(trainingData)):
        dist = euclideanDistance(test, trainingData[x], length)
        distance_measure.append(( dist, trainingData[x]))

    distance_measure.sort()

    neighbors =[]
    for x in range(k):
        neighbors.append(distance_measure[x][1])
    return neighbors



def determineClass(neighbors):
    classMajority = {}
    for x in range(len(neighbors)):
        classification = neighbors[x][-1]
        if classification in classMajority:
            classMajority[classification] +=1
        else:
            classMajority[classification] = 1

    lis = []
    for x in classMajority:
        lis.append(x)


    xis= []
    for x in classMajority.values():
       xis.append(x)


    max = -1
    counter = 0
    max_counter=0
    for y in xis:
        counter +=1
        if(y>max):
            max =y
            max_counter = counter


    num = ''
    for x in range(len(lis)):
        if x == max_counter-1:
            # print lis[x]
            num = lis[x]

    return num



D = [[170, 57, 'W'],
[192, 95, 'M'],
[150, 45, 'W'],
[170, 65, 'M'],
[175, 78, 'M'],
[185, 90, 'M'],
[170, 65, 'W'],
[155, 48, 'W'],
[160, 55, 'W'],
[182, 80, 'M'],
[175, 69, 'W'],
[180, 80, 'M'],
[160, 50, 'W'],
[175, 72, 'M']]

# Taking input from user
data_point = input('Enter the data point, separeted with a comma: Height and Weight \n')

data_point = list(data_point)

X = []
for x in data_point:
    X.append(x)

k = input('Enter an int value for k \n')


result = checkNeighbors(D, X, k)

result1 = determineClass(result)
print 'The predicted class is: ' + result1


