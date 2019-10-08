'''Shreyash Shrivastava
1001397477'''


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

D = { ((170, 57, 32),'W'),
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
((175, 72, 30), 'M'), }

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

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
        classification = 'Probability: '+ prob.__str__() + '\nClassified as a Woman\n'
        if prob >0.5: classification='Probability:'  + prob.__str__() + '\nClassified as a Man\n'
        return classification

    def sigmoid(self,ws): return 1.0/(1+np.exp(-ws))

logicReg = LogisticRegresion()
warray = logicReg.gradient_desent(np.mat(xarray),np.mat(yarray).transpose())
print('Data Set: '+ '[155,40,35] ' +logicReg.classify([155,40,35],warray))
print('Data Set: '+ '[170,70,32] ' +logicReg.classify([170,70,32],warray))
print('Data Set: '+ '[175,70,35] ' +logicReg.classify([170,75,35],warray))
print('Data Set: '+ '[180,90,20] ' +logicReg.classify([180,90,20],warray))

