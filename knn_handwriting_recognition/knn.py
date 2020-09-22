import numpy as np
import operator
from os import listdir
import matplotlib.pyplot as plt

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

def calDistance(vec1,vec2):
    vec_array1, vec_array2 = np.array(vec1), np.array(vec2)
    diff_array = np.abs(vec_array1 - vec_array2)
    return np.sum(np.average(diff_array))

def sortDictionary(dict):
    listDict = []
    for k in dict.keys():
        listDict.append([dict[k],k])
    listDict.sort(reverse=True)
    return listDict[0][1]

def generateDistanceData(directTesting, directTraining):
    dataTraining, classTraining = loadTrainingFile(directTraining)
    dataTesting, classTesting = loadTrainingFile(directTesting)
    sizeTest, sizeTrain = len(classTesting), len(classTraining)
    distanceList = np.zeros((sizeTest,sizeTrain,2))   
    for i in range(len(classTesting)):
        print('KNN running ',np.round(i/len(classTesting)*100,2),'%',end='\r')
        knnList = []
        for j in range(len(classTraining)):
            knnList.append([calDistance(dataTesting[i],dataTraining[j]),classTraining[j]])
        knnList.sort()
        distanceList[i,:,:] = np.array(knnList)
    return distanceList, classTesting

def classifyTestDigit(distanceList, classTesting, k):
    sizeTest = distanceList.shape[0]
    knnResult = []
    for i in range(sizeTest):
        knnList = distanceList[i,:,:]
        directClass = {}
        for m in range(k):
            if knnList[m][1] in directClass.keys():
               directClass[knnList[m][1]] += 1
            else:
                directClass[knnList[m][1]] = 1
        knnResult.append(sortDictionary(directClass))
    return knnResult

def printResult(knnResult, classTesting):
    errorCount = 0
    testSize = len(knnResult)
    for i in range(testSize):
        # print('Num:',classTesting[i],',testing result',knnResult[i])
        if knnResult[i] != classTesting[i]:
            errorCount += 1

    # print('error rate:',np.round(errorCount/testSize*100,2),'%')
    return errorCount/testSize

if __name__ == "__main__":
    directTraining = 'digits/trainingDigits/'
    directTesting = 'digits/testDigits/'

    # distanceList, classTesting = generateDistanceData(directTesting, directTraining)
    # np.save('distanceList.npy',distanceList)
    # np.save('classTesting.npy',classTesting)
    classTesting = np.load('classTesting.npy')
    distanceList = np.load('distanceList.npy')

    k_list = np.array([i for i in range(1,21)])
    error_rate = []
    for k in k_list:
        knnResult = classifyTestDigit(distanceList, classTesting,k)
        error_rate.append(printResult(knnResult, classTesting)*100)
    plt.figure(figsize=(5,3))
    plt.plot(k_list,error_rate,'b')
    plt.ylabel('error rate *100 %')
    plt.xlabel('k')
    plt.xlim([0,20])
    plt.xticks(k_list, k_list)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.16)
    plt.savefig('error_rate.pdf')

























