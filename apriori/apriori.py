    

class ApriopriCT(object):

    def __init__(self, dataList, itemList, minSupport=0.1, minConfidence=0.6):
        self.dataList = dataList
        self.minSupport = minSupport
        self.minConfidence = minConfidence
        self.itemList = itemList
        self.processedData = []
        self.supportDictionary = dict()
        self.confidenceDictionary = dict()
        self.C0 = self.C0 = [[i] for i in range(len(self.itemList))]
        self.Ck = []
        self.Rules = []
        self.prepareData()

    def prepareData(self):
        processedData = []
        for raw in self.dataList:
            temp = []
            for item in self.itemList:
                if item in raw:
                    temp.append(1)
                else:
                    temp.append(0)
            self.processedData.append(temp)
        return

    def match(self, items, dataL):
        for i in items:
            if dataL[i] == 0:
                return False
        return True

    def updateC(self, C):
        c_update = []
        for ci in C:
            count = 0
            for raw in self.processedData:
                if self.match(ci, raw):
                    count += 1
            if count > len(self.dataList)*self.minSupport:
                c_update.append(ci)
                # print(ci)
                self.supportDictionary[frozenset(ci)] = count/len(self.dataList)
        return c_update

    def isApriori(self,l,C):
        for i in range(len(l)):
            subList = l[:i]+l[i+1:]
            if subList not in C:
                return False
        return True

    def create_Ck(self,C,k):
        Ck = []
        lenC = len(C)
        for i in range(lenC):
            for j in range(i+1,lenC):
                l1, l2 = C[i], C[j]
                l1.sort()
                l2.sort()
                if l1 ==[] or l1[:-1] == l2[:-1]:
                    l = l1.copy()
                    l.append(l2[-1])
                    if i == 1 or self.isApriori(l, C):
                        Ck.append(sorted(l))
        return Ck

    def findFreqList(self):
        C = self.C0.copy()
        C = self.updateC(C)
        i = 1
        while i<len(self.itemList)+1:
            updateC = self.create_Ck(C,i)
            Ck = self.updateC(updateC)
            if len(Ck) == 0:
                break
            else:
                C = Ck
            i+=1
        self.Ck = C.copy()
        return

    def printFreqList(self):
        for k in self.Ck:
            combo = [self.itemList[i] for i in k]
            print('combo:',combo,',freq:', round(self.supportDictionary[frozenset(k)],5))
        # for k in self.supportDictionary.keys():
        #     combo = [self.itemList[i] for i in k]
        #     print('combo:',k,',freq:', round(self.supportDictionary[k],5))
        return

    def calConfidence(self,set1,set2):
        ratio = self.supportDictionary[set1]/self.supportDictionary[set2]
        return ratio

    def buildRules(self):
        freqList = self.Ck.copy()
        # for combo in freqList:
        #     for item in combo:
                
        return

    def run(self):
        self.findFreqList()
        self.printFreqList()


if __name__ == '__main__':
    f =  open('MBA_data.txt').readlines()
    dataList = []
    for x in f:
        L = x.split()
        dataList.append(L) 
    itemSet = set()
    for raw in dataList:
        for l in raw:
            itemSet.add(l)
    itemList = sorted(list(itemSet),reverse=False)

    apriSolution = ApriopriCT(dataList,itemList)
    apriSolution.run()























