#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

class TreeNode(object):
    def __init__(self, x):
        self.depth = x
        self.left = None
        self.right = None

class CARTmethod(object):
    def __init__(self,dataList ,max_depth,min_size):
        self.dataList = dataList
        self.max_depth = max_depth
        self.min_size = min_size
        self.myTree = TreeNode(0)
        self.subListCollection = []

    def split(self,dataList,index):
        yt = dataList[index][1]
        # subList1 = dataList[index:]
        # subList2 = dataList[:index]
        subList1, subList2 = [], []
        for i in range(len(dataList)):
            if dataList[i][1]<yt:
                subList1.append(dataList[i])
            else:
                subList2.append(dataList[i])
        return subList1, subList2

    def linearRegression(self,x,y):
        n = len(x)
        xy = [x[i]*y[i] for i in range(n)]
        x2 = [x[i]**2 for i in range(n)]
        b1 = (sum(xy)-sum(x)*sum(y)/n)/(sum(x2)-1/n*sum(x)**2)
        b0 = (sum(y)-b1*sum(x))/n
        return b1, b0

    def calGiniIndex(self,dataList):
        x = [i[0] for i in dataList]
        y = [i[1] for i in dataList]
        b1, b0 = self.linearRegression(x,y)
        err = [(y[i]-x[i]*b1-b0)**2 for i in range(len(x))]
        return sum(err)

    def findBestSplit(self,dataList):
        y = [i[1] for i in dataList]
        out, outList1, outList2 = -1, [], []
        gini = float('inf')
        for i in range(len(dataList)):
            subList1,subList2 = self.split(dataList,i)
            if len(subList1)<self.min_size or len(subList2)<self.min_size:
                continue
            ginisplit = self.calGiniIndex(subList1)+self.calGiniIndex(subList2)
            if ginisplit < gini:
                gini = ginisplit
                out = i
                outList1, outList2 = subList1,subList2

        return out, outList1, outList2

    def add_tree_node(self,myTree, dataList):
        if myTree.depth == self.max_depth:
            myTree.left = dataList
            return
        out, outList1, outList2 = self.findBestSplit(dataList)
        if out == -1:
            myTree.left = dataList
            return
        myTree.left = TreeNode(myTree.depth+1)
        myTree.right= TreeNode(myTree.depth+1)

        self.add_tree_node(myTree.left,outList1)
        self.add_tree_node(myTree.right,outList2)

    def getSubListCollection(self,myTree):
        if isinstance(myTree.left,list):
            self.subListCollection.append(myTree.left)
            return
        if isinstance(myTree.right,list):
            self.subListCollection.append(myTree.right)
            return
        if myTree.left:
            self.getSubListCollection(myTree.left)
        if myTree.right:
            self.getSubListCollection(myTree.right)
        return

    def printCART(self):
        plt.figure(figsize=(10,6))
        for sublist in self.subListCollection:
            x = [i[0] for i in sublist]
            y = [i[1] for i in sublist]
            b1, b0 = self.linearRegression(x,y)
            yd = [xi*b1+b0 for xi in x]
            plt.plot(x,y,'ro',ms=1)
            plt.plot(x,yd,'g')
            plt.xlim([self.dataList[0][0],self.dataList[-1][0]])
            plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        plt.savefig('fitting_cart.pdf')
        return

    def run(self):
        self.add_tree_node(self.myTree, self.dataList)
        self.getSubListCollection(self.myTree)
        self.printCART()

def type1Data():
    x, y = [], []
    for j in range(5):
        x1 = [i/10+j*10 for i in range(100)]
        y1 = [0.5*(i-j*10)+20*j+np.random.normal(2,size=None) for i in x1]
        x += x1
        y += y1
    return x, y

def type2Data():
    x = [i/10 for i in range(1000)]
    y = [0.5*(xi-10*int(xi/10))+np.random.normal(0.5,size=None) for xi in x]
    return x, y

if __name__ == '__main__':
    x, y = type1Data()
    # plt.clf()
    # plt.plot(x,y,'ro'.ms=1)
    # plt.show()
    # quit()
    dataList = [[x[i],y[i]] for i in range(len(x))]
    maxDepth, miniSize = 10, 80
    cart = CARTmethod(dataList, maxDepth, miniSize)
    cart.run()

















