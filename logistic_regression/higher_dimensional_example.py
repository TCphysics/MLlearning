import numpy as np
import matplotlib.pyplot as plt
import logistic_regression

class LinearInseparableData(object):

    def __init__(self,N, distance, Ntime, dl):
        self.N = N
        self.distance = distance
        self.Ntime = Ntime
        self.dl = dl
        self.xList = []
        self.Color = []
        self.xList_r = []
        self.xList_4d = []
        self.category = []

    def generating2dData(self):
        xList = []
        category = []
        for i in range(N):
            x = np.random.normal(0,self.distance,size=None)
            y = np.random.normal(0,self.distance,size=None)
            r = np.sqrt(x**2+y**2)
            xList.append([x,y])
            if r < self.distance*1.2:
                category.append(0)
            else:
                category.append(1)
        return xList, category

    def randomWalk(self, xList):
        for j in range(self.Ntime):
            random_x = np.random.normal(0,self.dl,size=self.N)
            random_y = np.random.normal(0,self.dl,size=self.N)
            for i in range(self.N):
                xList[i][0] += random_x[i]
                xList[i][1] += random_y[i]
        return xList

    def plotData(self, xList,category):
        x, y = [],[]
        for i in range(self.N):
            x.append(xList[i][0])
            y.append(xList[i][1])
            if category[i] == 1:
                self.Color.append('red')
            else:
                self.Color.append('blue')
        plt.figure(figsize=(6,6))
        plt.scatter(x,y,color=self.Color,s=1)
        plt.xlim([-3*d,3*d])
        plt.ylim([-3*d,3*d])
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
        plt.savefig('Data_size='+str(self.N)+'.pdf')
        return

    def export4dData(self,xList):
        res = []
        for i in range(self.N):
            x,y = xList[i][0], xList[i][1]
            res.append([x,y,x**2,y**2])
        return res

    def run(self):
        self.xList, self.category = self.generating2dData()
        self.xList_r = self.randomWalk(self.xList)
        self.xList_4d = self.export4dData(self.xList_r)
        self.plotData(self.xList_r,self.category)

def hyperbolicFunction(x,t):
    res = 0
    for i in range(len(x)):
        res += x[i]*t[i]
    return res

def boundaryLine(t):
    L0 = np.array([2*i*np.pi/200 for i in range(200)]) 
    xc, yc = -t[0]/(2*t[2]), -t[1]/(2*t[3])
    a = t[0]**2/(4*t[2])+t[1]**2/(4*t[3])-t[4]
    xr, yr = np.sqrt(a/t[2]), np.sqrt(a/t[3])
    x = xc + np.cos(L0) * xr
    y = yc + np.sin(L0) * yr
    return x, y 

def plotResult(xList):
    xMat = np.array(xList)
    L0 = np.array([2*i*np.pi/200 for i in range(200)]) 
    plt.figure(figsize=(6,6))
    plt.scatter(xMat[:,0],xMat[:,1],color=colorLine,s=1)
    plt.plot(xBound, yBound,'k',markersize=0.5,label='Solution Boundary')
    plt.plot(d*np.cos(L0),d*np.sin(L0),'g',ms=0.5,label='Initial Boundary')
    plt.xlim([-3*d,3*d])
    plt.ylim([-3*d,3*d])
    plt.legend()
    plt.title('N='+str(N)+', cycleMax='+str(cycleMax)+', stepSize='+str(stepSize))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    plt.savefig('4DSolution_Ncyc='+str(cycleMax)+'_dl='+str(stepSize)+'.pdf')
    return

def calErrorRate(xList,Category,theta):
    count = 0
    for i in range(2*N):
        sign = np.sign(theta[0] * xList[i][0] + theta[1] * xList[i][1] + theta[2])
        sign = int(0.5*sign+0.5)
        # print(sign, Category[i])
        if sign != Category[i][0]:
            count+=1
    return count/(2*N)

N, d, Ntime, dl = 1000, 10, 100, 0.1
initialState = LinearInseparableData(N, d, Ntime, dl)
initialState.run()

xList = initialState.xList_4d # [x, y, x2, y2]
category = initialState.category
colorLine = initialState.Color

cycleMax, stepSize = 50000, 0.05
LRsolution = logistic_regression.LogisticRegression(xList, category, cycleMax, stepSize)
theta_solu = LRsolution.run()
xBound, yBound = boundaryLine(theta_solu)
plotResult(xList)














