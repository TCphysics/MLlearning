import operator
from os import listdir
import numpy as np

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


dataTraining, classTraining = loadTrainingFile(directTraining)
dataTesting, classTesting = loadTrainingFile(directTesting)
sizeTest, sizeTrain = len(classTesting), len(classTraining)

print(classTraining)