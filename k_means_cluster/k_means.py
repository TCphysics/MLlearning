import numpy as np
import matplotlib.pyplot as plt

class KMSolution(object):

    def __init__(self,dataList,knum,Niteration):
        self.dataList = dataList
        self.knum = knum
        self.Niteration = Niteration
        self.centerList = []
        self.clusterList = [-1 for i in range(len(self.dataList))]

    def initializeCenterList(self,x_range,y_range):
        for i in range(self.knum):
            xc = np.random.uniform(low=x_range[0], high=x_range[1], size=None)
            yc = np.random.uniform(low=y_range[0], high=y_range[1], size=None)
            self.centerList.append([xc,yc])

    def calDistance(self,p1,p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def initialization(self):
        x = [p[0] for p in self.dataList]
        y = [p[1] for p in self.dataList]
        x_range = [min(x),max(x)]
        y_range = [min(y),max(y)]
        self.initializeCenterList(x_range,y_range)
        return

    def updating_centering(self):
        for i in range(len(self.dataList)):
            sqd_min = 1e10
            for j in range(self.knum):
                sqd = self.calDistance(self.dataList[i], self.centerList[j])
                if sqd < sqd_min:
                    sqd_min = sqd
                    self.clusterList[i] = j
        centerCount = self.clusterList.count(set(self.clusterList))
        for j in range(self.knum):
            x_cl, y_cl = [], []
            xc, yc = 0, 0
            for i in range(len(self.dataList)):
                if self.clusterList[i] == j:
                    x_cl.append(self.dataList[i][0])
                    y_cl.append(self.dataList[i][1])
            if len(x_cl) > 0:
                xc = sum(x_cl)/len(x_cl)
                yc = sum(y_cl)/len(y_cl)
                self.centerList[j] = [xc, yc]

    def plotResult(self,para):
        x = [p[0] for p in self.dataList]
        y = [p[1] for p in self.dataList]
        xc = [p[0] for p in self.centerList]
        yc = [p[1] for p in self.centerList]
        plt.figure(figsize=(8,8))
        plt.plot(x,y,'ro',ms=2)
        plt.plot(xc,yc,'bo',ms=5)
        plt.xlim([-40,40])
        plt.ylim([-40,40])
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        plt.savefig(para+'.pdf')
        return
    def run(self):
        self.initialization()
        for k in range(self.Niteration):
            print(np.round(k/self.Niteration*100,1),'%',end='\r')
            self.updating_centering()

def generatingData(Size,centerNum,radius):
    theta = [2*i*np.pi/centerNum for i in range(centerNum)]
    d = 2*np.pi*radius/centerNum
    x0,y0 = [], []
    N = Size//centerNum
    for i in range(centerNum):
        for j in range(N):
            x0.append(np.random.normal(loc=radius*np.cos(theta[i]),scale=d/7,size=None))
            y0.append(np.random.normal(loc=radius*np.sin(theta[i]),scale=d/7,size=None))
    return x0, y0

if __name__ == '__main__':
    dataSize, centerNum, radius = 500, 10, 20
    x, y = generatingData(dataSize, centerNum, radius)

    dataList = [[x[i], y[i]] for i in range(len(x))]
    knum, Niteration = 15, 2000
    km_solution = KMSolution(dataList,knum,Niteration)
    km_solution.run()

    cluster = km_solution.clusterList
    centerList = km_solution.centerList
    Count = [cluster.count(i) for i in range(knum)]
    centerList = np.array([centerList[i] for i in range(knum) if Count[i]>0])
    print(centerList.shape)
    plt.figure(figsize=(8,8))
    plt.plot(x,y,'ro',ms=1)
    plt.plot(centerList[:,0],centerList[:,1],'bo',ms=2.5)
    plt.xlim([-2*radius,2*radius])
    plt.ylim([-2*radius,2*radius])
    plt.savefig('cluster_k='+str(knum)+'.pdf')
























