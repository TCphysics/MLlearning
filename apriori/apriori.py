    
class ApriopriCT(object):
    '''
    This python script finds the frequency list FL with maximum length and
    then build rules based on each list in FL.
    '''


    def __init__(self, dataList, itemList, minSupport=0.2, minConfidence=0.8):
        
        '''
        param dataList: original dataList
        param minSupport: minimum Support
        param minConfidence: minimum Confidence
        param itemList: list of all items
        param processedData: processed Data with 1 and 0 in each raw representing
                             if items are included.
        param supportDictionary: Dictionary of support
        param confidenceList: List of confidence
        param Ck: optimal frequency list depending on minSupport
        '''
        self.dataList = dataList
        self.minSupport = minSupport
        self.minConfidence = minConfidence
        self.itemList = itemList
        self.processedData = []
        self.supportDictionary = dict()
        self.confidenceList = []
        self.C0 = self.C0 = [[i] for i in range(len(self.itemList))]
        self.freqList = []
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
        # decide if a items combo is contained in a single data
        for i in items:
            if dataL[i] == 0:
                return False
        return True

    def updateC(self, C):
        # Drop effective combo when its freq lower than minSuport
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
        # decide if a Ck have sublist outside the Ck-1.
        for i in range(len(l)):
            subList = l[:i]+l[i+1:]
            if subList not in C:
                return False
        return True

    def create_Ck(self,C,k):
        #generate Ck based on Ck-1
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
        # find the frequency list
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
        self.freqList = C.copy()
        return

    def printFreqList(self):
        for k in self.freqList:
            combo = [self.itemList[i] for i in k]
            print('Freq list:',combo,',freq:', round(self.supportDictionary[frozenset(k)],5))
        # for k in self.supportDictionary.keys():
        #     combo = [self.itemList[i] for i in k]
        #     print('combo:',k,',freq:', round(self.supportDictionary[k],5))
        return

    def calConfidence(self,l,freqCombo):
        #calculate Conf of a sublist based on a combo
        set1 = frozenset(freqCombo)
        set2 = frozenset(freqCombo) - frozenset(l)
        ratio = self.supportDictionary[set1]/self.supportDictionary[set2]
        return ratio

    def updateCr(self, Cr, freqCombo):
        # drop items list if its conf is lower han minConfidence
        c_update = []
        for c in Cr:
            if self.calConfidence(c, freqCombo) > self.minConfidence:
                c_update.append(c)
        return c_update

    def findRule(self, freqCombo):
        # build rule list for a combo
        Cr = [[f]  for f in freqCombo]
        i = 1
        ruleList = []
        while i < len(freqCombo):
            updateCr = self.updateCr(Cr,freqCombo)
            ruleList += updateCr
            if len(updateCr) == 0:
                break
            Cr = self.create_Ck(updateCr,i)
            i+=1
        return ruleList

    def addRuleToConfDict(self,ruleList, freqCombo):
        # find rule and fulfill confidenceList for a freqCombo
        for rule in ruleList:
            conf = self.calConfidence(rule, freqCombo)
            tag = [rule, list(frozenset(freqCombo)-frozenset(rule))]
            self.confidenceList.append( [tag, conf] )

    def buildRules(self):
        # iterates over each combo in freqList.
        freqList = self.freqList.copy()
        for combo in freqList:
            # print('combo:',combo)
            ruleList = self.findRule(combo)
            self.addRuleToConfDict(ruleList, combo)
            # print('ruleList:',ruleList)
        return

    def printRules(self):
        for k in self.confidenceList:
            r1 = [self.itemList[i] for i in k[0][1]]
            r2 = [self.itemList[i] for i in k[0][0]]
            print('rule:',r1,' to ',r2 ,', conf=',round(k[1],4))

    def run(self):
        self.findFreqList()
        self.printFreqList()
        self.buildRules()
        self.printRules()


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























