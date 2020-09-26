# -*- coding: UTF-8 -*-
from math import log
import operator


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['age', 'job', 'house', 'credit']        #分类属性
    return dataSet, labels

def calcShannonEnt(dataSet):
    sizeData = len(dataSet)                        #返回数据集的行数
    labelCounts = {}                                 #保存每个标签(Label)出现次数的字典
    for raw in dataSet:                          #对每组特征向量进行统计
        currentLabel = raw[-1]                   #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():   #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1               #Label计数
    shannonEnt = 0.0                                 #经验熵(香农熵)
    for key in labelCounts:                          #计算香农熵
        prob = float(labelCounts[key]) / sizeData  #选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)            #利用公式计算
    return shannonEnt                                #返回经验熵(香农熵)

def reducedDataSet(dataSet, labelPosition, labelValue):
    rDataSet = []                                     #创建返回的数据集列表
    for raw in dataSet:                             #遍历数据集
        if raw[labelPosition] == labelValue:
            reducedRaw = raw[:labelPosition]             #去掉axis特征
            reducedRaw.extend(raw[labelPosition+1:])     #将符合条件的添加到返回的数据集
            rDataSet.append(reducedRaw)
    return rDataSet                                   #返回划分后的数据集

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                     #特征数量
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                    #信息增益
    bestFeature = -1                                      #最优特征的索引值
    for i in range(numFeatures):                          #遍历所有特征
        #获取dataSet的第i个所有特征
        featureList = [raw[i] for raw in dataSet]
        featureSet = set(featureList)                         #创建set集合{},元素不可重复
        newEntropy = 0.0                                   #经验条件熵
        for value in featureSet:                           #计算信息增益
            subDataSet = reducedDataSet(dataSet, i, value)           #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)        #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                        #信息增益
        # print("The %d th feature gains entropy %.3f" % (i, infoGain))             #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                              #计算信息增益
            bestInfoGain = infoGain                                #更新信息增益，找到最大的信息增益
            bestFeature = i                                        #记录信息增益最大的特征的索引值
    return bestFeature                                             #返回信息增益最大的特征的索引值

def majorityCnt(labelList):
    classCount = {}
    for vote in labelList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0] 

"""
函数说明:递归构建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""
def createTree(dataSet, labels, featLabels):
    labelList = [raw[-1] for raw in dataSet]
    '''
    When the sub data set has homogeneous label (perfect classification!), we just
    need to return the label
    '''
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    '''
    At the end of a branch when only feature is last (num of column =1). 
    We choose the label with majority count. 
    '''
    if len(dataSet[0]) == 1:
        return majorityCnt(labelList)


    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureName = labels[bestFeature]
    featLabels.append(bestFeatureName)
    myTree = {bestFeatureName:{}}
    del(labels[bestFeature])# Delete the selected feature from labelList
    featureValue = [raw[bestFeature] for raw in dataSet]
    for value in set(featureValue):
        subLabels=labels[:]
        # recrusion
        myTree[bestFeatureName][value] = createTree(reducedDataSet(dataSet, bestFeature, value), subLabels, featLabels)
    return myTree


"""
函数说明:使用决策树执行分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVector - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""
def classify(inputTree, featureLabels, testVector):
    firstStr = next(iter(inputTree))             #获取决策树结点
    secondDict = inputTree[firstStr]             #下一个字典
    featIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVector[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    testVec = [0, 1]     # 测试数据
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('good credit')
    if result == 'no':
        print('low credit')