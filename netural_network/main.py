import operator
from os import listdir
import numpy as np
from network_lib import netural_network

def convertFilet2Vector(filename):
    file = open(filename)
    dataVec = []
    while file.readline():
        line = file.readline()
        for j in list(line):
            if j != '\n':
                dataVec.append(int(j))
    return dataVec

def loadTrainingFile(directTraining):
    testFileList = listdir(directTraining)
    trainingSize = len(testFileList)
    dataVectorCollection = []
    classNum = []
    for i in range(trainingSize):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum.append(int(fileStr.split('_')[0]))
        dataVectorCollection.append(convertFilet2Vector(directTraining+fileNameStr))
    return dataVectorCollection, classNum

directTraining = 'digits/trainingDigits/'
directTesting = 'digits/testDigits/'

print('Loading data...')
dataTraining, classTraining = loadTrainingFile(directTraining)
dataTesting, classTesting = loadTrainingFile(directTesting)
sizeTest, sizeTrain, sizeData = len(classTesting), len(classTraining), len(dataTraining[0])

print('Data loaded')
Niteration = 5000
inData = dataTraining.copy()
testInData = dataTesting.copy()
outData, testOutData = [], []
for num in classTraining:
    outData.append([1 if i == num else 0 for i in range(10)])
for num in classTesting:
    testOutData.append([1 if i == num else 0 for i in range(10)])

print('Netral work program starts!')
NNlayers = [200, 100, 50]
Sizes = [sizeData] + NNlayers + [10]

Weights = [np.random.random((y,x)) for x,y in zip(Sizes[:-1], Sizes[1:])]
Bias = [np.random.random((x,1)) for x in Sizes[1:]]

i, count, cmax = 0, 0, 0
for j in range(Niteration*sizeTrain):
    cmax = max(count, cmax)
    if count > 1000:
        break
    NN = netural_network(NNlayers, inData[i], outData[i], Weights, Bias)
    NN.run()
    if NN.error > 1e-3:
        count = 0
    else:
        count += 1    
    print('cmax:',cmax,', count:',count,',',np.round(j/Niteration/sizeTrain*1e2, 2),'% ',end='\r')
    Weights, Bias = NN.Weights, NN.Bias
    i = (i+1)%sizeTrain

print('Netral work program ends!')
Error = 0

for i in range(sizeTest):
    NNtest = netural_network(NNlayers, testInData[i], testOutData[i], Weights, Bias)
    NNtest.forward_propagation()
    error = np.sum(np.square(NNtest.A[-1] - NNtest.y))
    if error >1e-3:
        Error += 1/sizeTest
    del(NNtest)

print(Error)








