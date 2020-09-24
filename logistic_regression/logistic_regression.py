import numpy as np

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
        catVec = np.array(self.trainingCategory).reshape(len(self.trainingCategory),1)
        trainMat = self.trainingData.copy()
        for i in range(len(trainMat)):
            trainMat[i].append(1)

            
        return np.array(trainMat), catVec

    def Sigmoid(self, xMat, thetaVec):
        res = np.zeros(xMat.shape[0], dtype=np.float128)
        for i in range(xMat.shape[0]):
            res[i] = np.dot(xMat[i,:],thetaVec)

        res_0 = 1/(1+np.exp(-res))
        return res_0

    def iteration(self, trainingMat, catVec):
        xMat = trainingMat.copy()
        thetaVec = np.ones(len(self.trainingData[0]))
        for c in range(self.cycleMax):
            print(np.round(100*c/self.cycleMax,2),'%',end='\r')
            E = self.Sigmoid(xMat, thetaVec)
            E = E.reshape(len(self.trainingCategory),1)
            delta = self.stepSize/xMat.shape[0] * (E-catVec)
            delta = np.dot(np.transpose(delta),xMat)
            # print('delta:',delta.shape)
            thetaVec -= delta.reshape(len(xMat[0]))
        return thetaVec

    def run(self):
        trainMat, catVec = self.buildMatrix()
        theta_solu = self.iteration(trainMat, catVec)
        return theta_solu

    def plotResult(self, theta_solu):
        return

# if __name__ == "__main__":



