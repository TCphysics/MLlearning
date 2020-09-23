import numpy as np

def loadData(self, directory):
    '''
    :param directory: directory of data
    '''
    return

class LogisticRegression(object):
    '''
    Logistic regression python code.
    '''
    def __init__(self,trainingData, trainingCategory, cycleMax, stepSize):
        '''
        :param trainingData: input x matrix(N*M) in list.
        :param trainingCategory: input y vector (M*1) in list.
        :param cycleMax: the num of LR iteration times.
        :param stepSize: iteraion step length <<1.
        '''
        self.trainingData = trainingData
        self.trainingCategory = trainingCategory
        self.cycleMax = cycleMax
        self.stepSize = stepSize

    def buildMatrix(self):
        trainMat = self.trainingData.copy()
        for i in range(len(trainMat)):
            trainMat[i].append(1)
        return np.array(trainMat), np.array(self.trainingCategory)

    def Sigmoid(self, xMat, thetaVec):
        res = np.zeros(xMat.shape[0])
        for i in range(xMat.shape[0]):
            res[i] = np.dot(xMat[i,:],thetaVec)
        res = 1/(1+np.exp(res))
        return res

    def iteration(self, xMat, catVec):
        thetaVec = np.ones(trainingData)
        for c in range(cycleMax):
            E = self.Sigmoid(xMat, thetaVec)
            delta = self.stepSize/xMat.shape[0] * (E-catVec)
            thetaVec -= delta
        return thetaVec

    def run(self):
        trainMat, catVec = self.buildMatrix()
        theta_solu = self.iteration(self, xMat, catVec)
        return theta_solu

    def plotResult(self, theta_solu):
        return

if __name__ == "__main__":


























