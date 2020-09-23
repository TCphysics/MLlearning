import numpy as np
import matplotlib.pyplot as plt
import logistic_regression

def generating2dData(N,distance):
    xList = []
    category = []
    for i in range(N):
        x = np.random.normal(0,distance/2.5,size=None)
        y = np.random.normal(0,distance/2.5,size=None)
        xList.append([x,y])
        category.append([0])

    for i in range(N):
        x = np.random.normal(distance,distance/2.5,size=None)
        y = np.random.normal(distance,distance/2.5,size=None)
        xList.append([x,y])
        category.append([1])
    return xList, category

def plotResult(xList,theta):
    xMat = np.array(xList)
    xaxis = np.linspace(-d,2*d,1000)
    yaxis = -1/theta[1] * (theta[0] * xaxis + theta[2])
    y0 = d - xaxis
    plt.figure(figsize=(6,6))
    plt.plot(xMat[:N,0],xMat[:N,1],'ro',markersize=1)
    plt.plot(xMat[N:,0],xMat[N:,1],'bo',markersize=1)
    plt.plot(xaxis,yaxis,'g',linewidth = 1,markersize=0.5,label='solution line, error rate ='+str(errorRate)+'%')
    plt.plot(xaxis,y0,'grey',linewidth = 0.3,markersize=0.3,label='intuitive line')
    plt.xlim([-d,2*d])
    plt.ylim([-d,2*d])
    plt.legend()
    plt.title('N='+str(2*N)+',cycleMax='+str(cycleMax)+',stepSize='+str(stepSize))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08)
    plt.savefig('Solution_Ncyc='+str(cycleMax)+'_dl='+str(stepSize)+'.pdf')
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

N, d = 250, 10
xList, cat = generating2dData(N, d)
cycleMax, stepSize = 20000, 0.1
LRsolution = logistic_regression.LogisticRegression(xList, cat, cycleMax, stepSize)
theta_solu = LRsolution.run()
errorRate  = np.round(calErrorRate(xList,cat,theta_solu)*100,2)
print('error rate:',errorRate)
plotResult(xList, theta_solu)
















