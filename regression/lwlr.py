import numpy as np
import matplotlib.pyplot as plt
import random

class localWeightedLinearRegression(object):

    def __init__(self, xInput, yInput, k=0, xTest=0):
        self.xInput = xInput
        self.yInput = yInput
        self.k = k
        self.xTest = xTest
        self.X = list()
        self.Y = list()
        self.W = None
        self.V = None
        # self.WL = list()

    def remodelInput(self):
        for x in self.xInput:
            if isinstance(x, list):
                self.X.append([1].extend(x))
            else:
                self.X.append([1,x])

        self.X = np.array(self.X)
        self.Y = (np.array(self.yInput)).reshape(len(self.xInput),1)

    def LRsolution(self):
        XTX = np.dot(np.transpose(self.X), self.X)
        XTY = np.dot(np.transpose(self.X), self.Y)
        self.W = np.dot(np.linalg.inv(XTX), XTY)

    def weightFunction(self):
        temp = []
        if isinstance(self.xInput[0], list):
            for x in self.xInput:
                d = sum([(x[i]-self.xTest[i])**2 for i in range(len(self.xInput[0]))])

                temp.append(np.exp(-d/(2*self.k)))
        else:
            for x in self.xInput:
                temp.append(np.exp(-(x-self.xTest)**2/(2*self.k)))

        self.V = np.mat(np.diag(temp))

    def LWLRsolution(self):
        self.X = np.mat(self.X)
        XTVX = self.X.T*(self.V * self.X)
        XTVY = self.X.T * (self.V * self.Y)
        self.W = XTVX.I * XTVY
        self.W = np.asarray(self.W)

    def run(self):
        self.remodelInput()
        if not self.k:
            self.LRsolution()
        else:
            self.weightFunction()
            self.LWLRsolution()
        return

if __name__ == '__main__':

    k = 10
    x = [i/10 for i in range(200)]
    y = [0.5*i +np.sin(i) + np.random.normal(0,0.2,size=None) for i in x]

    y0 = []
    for xi in x:
        # print(x.index(xi)/len(x)*100,end='\r')
        LWLR = localWeightedLinearRegression(x, y, k = k, xTest = xi)
        LWLR.run()
        W = LWLR.W
        yi = xi * W[1]+W[0]
        y0.append(yi[0])

    plt.figure(figsize=(8,5))
    plt.plot(x,y, 'ro',ms=2.5,label='original data')
    plt.plot(x,y0,'b' ,ms=1.5,label='LWLR fitting, k='+str(k))
    plt.legend()
    plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.1)
    plt.savefig('2d_lwlr_fitting_k='+str(k)+'.pdf')


































