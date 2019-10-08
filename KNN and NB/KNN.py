'''
    Shreyash Shrivastava
    1001397477
    CSE 6363

'''

import math

# Height Weight Age Class
D =  [[170, 57, 32, 'W'],
[192, 95, 28, 'M'],
[150, 45, 30, 'W'],
[170, 65, 29, 'M'],
[175, 78, 35, 'M'],
[185, 90, 32, 'M'],
[170, 65, 28, 'W'],
[155, 48, 31, 'W'],
[160, 55, 30, 'W'],
[182, 80, 30, 'M'],
[175, 69, 28, 'W'],
[180, 80, 27, 'M'],
[160, 50, 31, 'W'],
[175, 72, 30, 'M']]



# Cartesian distance calculator function
def euclideanDistance(item1, item2, length):
    cal_dist = 0
    for x in range(length):
        cal_dist += pow((item1[x] - item2[x]), 2)
    return math.sqrt(cal_dist)

# Function to check the nearest points from the given input
def checkNeighbors(trainingData, test, k):
    distance_measure =[]
    length = len(test)

    # calling the distance calculator function to measure each distance and add to a list
    for x in range(len(trainingData)):
        dist = euclideanDistance(test, trainingData[x], length)
        distance_measure.append(( dist, trainingData[x]))

    distance_measure.sort()
    for x in distance_measure:
        print(x)
    neighbors =[]
    for x in range(k):
        neighbors.append(distance_measure[x][1])
    return neighbors


# Function to see the majority class near the point and predict the class
def determineClass(neighbors):
    classMajority = {}

    # Determining the classes majority nearest to the given data point
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
            num = lis[x]
    return num


#X = [155, 40, 35]
# Taking input from user
data_point = input('Enter the data point, separeted with a comma: Height, Weight and age \n')
data_point = list(data_point)

X =[]
for x in data_point:
    X.append(x)


result = checkNeighbors(D, X, 3)

result_final = determineClass(result)

if result_final == 'W':
    print('Predicted as: Woman')
else:
    print('Predicted as: Man')




