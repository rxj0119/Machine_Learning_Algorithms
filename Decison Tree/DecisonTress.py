'''Shreyash Shrivastava
    1001397477'''

import math
import csv
import sys
class DecisionTree:
    def read_data(self, filename):
        fid = open(filename, "r")
        data = []
        d = []
        for line in fid.readlines():
            d.append(line.strip())
        for d1 in d:
            data.append(d1.split(","))
        fid.close()

        self.featureNames = ['Shape', 'Surface', 'Color', 'Bruises']
        data = data[1:]
        self.classes = self.get_classes(data)
        data = self.get_pure_data(data)
        return data, self.classes, self.featureNames

    def get_classes(self, data):
        data = data[0:]
        classes = []
        for d in range(len(data)):
            classes.append(data[d][0])
        return classes

    def get_features(self, data):
        features = data[0]
        #
        features = features[:-1]
        #
        return features

    def get_pure_data(self, dataRows):
        dataRows = dataRows[1:]
        for d in range(len(dataRows)):
            dataRows[d] = dataRows[d][1:]

        return dataRows

    def zeroList(self, size):
        d = []
        for i in range(size):
            d.append(0)
        return d

    def getArgmax(self, arr):
        m =0
        if len(arr) !=0:
            m = max(arr)
        ix = arr.index(m)
        return ix

    def getDistinctValues(self, dataList):
        distinctValues = []
        for item in dataList:
            if (distinctValues.count(item) == 0):
                distinctValues.append(item)
        return distinctValues

    def getDistinctValuesFromTable(self, dataTable, column):
        distinctValues = []
        for row in dataTable:
            if (distinctValues.count(row[column]) == 0):
                distinctValues.append(row[column])
        return distinctValues

    def getEntropy(self, p):
        if (p != 0):
            return -p * math.log2(p)
        else:
            return 0

    def create_tree(self, trainingData, classes, features, maxlevel=-1, level=0):
        nData = len(trainingData)
        nFeatures = len(features)

        try:
            self.featureNames
        except:
            self.featureNames = features

        newClasses = self.getDistinctValues(classes)
        frequency = self.zeroList(len(newClasses))
        totalEntropy = 0
        index = 0
        for aclass in newClasses:
            frequency[index] = classes.count(aclass)
            prob = float(frequency[index]) / nData
            totalEntropy += self.getEntropy(prob)
            index += 1

        default = classes[self.getArgmax(frequency)]
        if (nData == 0 or nFeatures == 0 or (maxlevel >= 0 and level > maxlevel)):
            return default
        elif classes.count(classes[0]) == nData:
            return classes[0]
        else:
            gain = self.zeroList(nFeatures)
            for feature in range(nFeatures):
                g = self.getGain(trainingData, classes, feature)
                gain[feature] = totalEntropy - g

            bestFeature = self.getArgmax(gain)

            newTree = {features[bestFeature]: {}}

            values = self.getDistinctValuesFromTable(trainingData, bestFeature)
            for value in values:
                newdata = []
                newClasses = []
                index = 0
                for row in trainingData:
                    if row[bestFeature] == value:
                        if bestFeature == 0:
                            newRow = row[1:]
                            newNames = features[1:]
                        elif bestFeature == nFeatures:
                            newRow = row[:-1]
                            newNames = features[:-1]
                        else:
                            newRow = row[:bestFeature]
                            newRow.extend(row[bestFeature + 1:])
                            newNames = features[:bestFeature]
                            newNames.extend(features[bestFeature + 1:])
                        newdata.append(newRow)
                        newClasses.append(classes[index])

                    index += 1

                subtree = self.create_tree(newdata, newClasses, newNames, maxlevel, level + 1)

                newTree[features[bestFeature]][value] = subtree
            return newTree


    def getGain(self, data, classes, feature):
        gain = 0
        ndata = len(data)-1

        values = self.getDistinctValuesFromTable(data, feature)
        featureCounts = self.zeroList(len(values))
        entropy = self.zeroList(len(values))
        valueIndex = 0
        for value in values:
            dataIndex = 0
            newClasses = []
            for row in data:
                if row[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])

                dataIndex += 1

            classValues = self.getDistinctValues(newClasses)
            classCounts = self.zeroList(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1

            for classIndex in range(len(classValues)):
                pr = float(classCounts[classIndex]) / sum(classCounts)
                entropy[valueIndex] += self.getEntropy(pr)

            pn = float(featureCounts[valueIndex]) / ndata
            gain = gain + pn * entropy[valueIndex]

            valueIndex += 1
        return gain

    def showTree(self, dic, seperator):
        if (type(dic) == dict):
            for item in dic.items():
                print(seperator, item[0])
                self.showTree(item[1], seperator + " | ")
        else:
            print(seperator + " -> (", dic + " )")

    def hasTree(self, dic,list_attrs,output_list):
        #list_attrs = {'Shape':'x','Surface':'s','Color':'n','Bruises':'t'}
        #print(dic)

        if (type(dic) == dict):
            for item in dic.items():
                if item[0] in list_attrs:
                    if list_attrs[item[0]] in item[1]:
                        if (item[1][list_attrs[item[0]]]) == 'e' or (item[1][list_attrs[item[0]]]) == 'p':
                            #print(item[1][list_attrs[item[0]]])
                            output_list.append(item[1][list_attrs[item[0]]])

                        else:
                            #print('g')
                            self.hasTree(item[1][list_attrs[item[0]]],list_attrs,output_list)
        else:
            #print(dic)
            output_list.append(dic)

        return output_list

tree = DecisionTree() # Decison Tree Object Created
tr_data, clss, attrs = tree.read_data(sys.argv[2])
# print(tr_data)
# print(clss)
# print(attrs)
tree1 = tree.create_tree(tr_data, clss, attrs)

#tree.showTree(tree1, '')
def testing_accuracy( dic, filename):
    t_count = 0
    correct_count = 0
    # Test Accuracy
    # Accuracy = amount of correct classifications / total amount of classifications
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            lis_attr = {'Shape':line[1],'Surface':line[2],'Color':line[3],'Bruises':line[4]}
            out = tree.hasTree(tree1,lis_attr,[])
            #print(out,line[5])
            outp = 'e'
            if len(out) >=1:
                outp = out[0]

            if outp == 'p' and str(line[5]) == 'p':
                correct_count+=1
            if outp == 'e' and str(line[5]) == 'n':
                correct_count +=1
            #print('Parameters->','Shape:', line[1], 'Surface:',line[2],'Color:',line[3],'Bruises:',line[4])
            #print('Expected: ',line[5])

            #print('Predicted:'), tree.hasTree(tree1, lis_attr)

            t_count+=1

    test_accu = correct_count/t_count
    return test_accu

def train_accuract(dic,filename):
    t_count = 0
    correct_count = 0
    # Train Accuracy
    # Accuracy = amount of correct classifications / total amount of classifications
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            lis_attr = {'Shape': line[1], 'Surface': line[2], 'Color': line[3], 'Bruises': line[4]}
            out = tree.hasTree(tree1, lis_attr, [])
            #print(out)
            # print(out,line[5])
            outp = out[0]

            if outp == 'p' and str(line[0]) == 'p':
                correct_count += 1
            if outp == 'e' and str(line[0]) == 'e':
                correct_count += 1
            # print('Parameters->','Shape:', line[1], 'Surface:',line[2],'Color:',line[3],'Bruises:',line[4])
            # print('Expected: ',line[5])

            # print('Predicted:'), tree.hasTree(tree1, lis_attr)

            t_count += 1
    #correct_count+=32
    test_accu = correct_count / t_count
    return test_accu


##// TO SHOW THE COMPLETE TREE, UNCOMMENT THE LINE BELOW
#tree.showTree(tree1,'') ##Uncomment to show the tree




# Test Accuracy
test_acc = testing_accuracy(tree1,sys.argv[1])
print('Test accuracy = ',test_acc)

# Train Accuracy
train_acc = train_accuract(tree1,sys.argv[2])
print('Train accuracy = ',train_acc)
# a= tree.hasTree(tree1,{'Shape':'x','Surface':'s','Color':'n','Bruises':'t'},[])
#
# print(a)