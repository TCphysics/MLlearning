import numpy as np
import pickle


class DecisionTree(object):

    def __init__(self,dataDirect, featreDirect):
        self.dataDirect = dataDirect
        self.featreDirect = featreDirect
        self.TitanicDataList = []
        self.featureInfo = []
        self.featureList = []
        self.DTree = {}

    def loadData(self):
        with open(self.dataDirect, 'rb') as filehandle:
            self.TitanicDataList = pickle.load(filehandle)

        with open(self.featreDirect, 'rb') as filehandle:
            self.featureInfo = pickle.load(filehandle)
        self.featureList = [x[0] for x in self.featureInfo]

    def calcShannonEnt(self, dataList):
        classlList = [row[-1] for row in dataList]
        prob = sum(classlList)/len(classlList)
        if prob == 1 or prob == 0:
            return 0
        Entropy = -1 * (prob*np.log(prob) + (1-prob)*np.log(1-prob) ) 
        return Entropy

    def splitListVertical(self, dataList, axialIndex, indexValue):
        reducedList = []
        for row in dataList:
            if row[axialIndex] == indexValue:
                reducedList.append(row)
        return reducedList

    def splitListHorizontal(self, dataList, axialIndex, value):
        reducedList = []
        for row in dataList:
            if row[axialIndex] == value:
                reducedRow = row[:axialIndex] + row[axialIndex+1:]
                reducedList.append(reducedRow)
        return reducedList

    def majorityCount(self, dataList):
        classlList = [row[-1] for row in dataList]
        label = 'Survived' if sum(classlList)>len(classlList)//2 else 'Died'
        return label

    def findBestFeatureToSplit(self, dataList):
        Entropy0 = self.calcShannonEnt(dataList)
        bestFeature = -1
        for i in range(len(dataList[0])-1):
            extraEntropy = 0
            featureVal = [row[i] for row in dataList]
            featureValSet = set(featureVal)
            for featV in featureValSet:
                subDataList = self.splitListVertical(dataList, i, featV)
                dEn = featureVal.count(featV)/len(featureVal)*self.calcShannonEnt(subDataList)
                extraEntropy += dEn
            if extraEntropy<Entropy0:
                Entropy0 = extraEntropy
                bestFeature = i
        return bestFeature

    def buildDecisionTree(self, dataList, featureList):
        classlList = [row[-1] for row in dataList]
        if classlList.count(classlList[0]) == len(classlList):
            return 'survived' if classlList[0]==1 else 'died'
        if not featureList:
            return self.majorityCount(dataList)

        bestFeature = self.findBestFeatureToSplit(dataList)
        beatFeatName = featureList[bestFeature]
        del(featureList[bestFeature]) # delete selected feature after every iteration
        myTree = {beatFeatName: {}}

        featureValue = [row[bestFeature] for row in dataList]
        featureValueSet = set(featureValue)
        for value in featureValueSet:
            subFeatureList = featureList.copy()
            subDataList = self.splitListHorizontal(dataList, bestFeature, value)
            myTree[beatFeatName][value] = self.buildDecisionTree(subDataList, subFeatureList)
            # print 'myTree', value, myTree
        return myTree


    def displayTree(self, Tree):
        return

    def run(self):
        self.loadData()
        self.DTree = self.buildDecisionTree( self.TitanicDataList, self.featureList)
        # print(self.DTree)
        return




if __name__ == '__main__':

    inputDirect = 'Titanic.data'
    featureDirect = 'Titanic_feature_list.data'

    decision_tree = DecisionTree(inputDirect, featureDirect)
    decision_tree.run()
    decisionTree = decision_tree.DTree
    # print(decisionTree)

























