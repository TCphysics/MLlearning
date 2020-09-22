import numpy as np
import _pickle as cPickle
import re
import os

class NBspamClassifier(object):
    '''
    This is python script containing naive Bayers method function for classifying
    spam email.
    Emails are categorized as 'spam' and 'ham'.
    25 spam and ham email are for history data. Approximately 5500
    emails are for testing.
    The smal size of history data results in high classifying error rate of
    13.4%. The spam email classifier is a tough job in the real life. Customized
    dynamic model is preferred now!
    '''

    def __init__(self, historyDataDirect, testDataDirect, classList):
        '''
        :param historyDataDirect: the training data input directory
        :param testDataDirect: the testing data input directory
        :param classList: category of email, there are two kinds now, and the
                        code is applicable to multiclass case. eg. recognize 
                        the author of a paper.
        '''
        self.historyDataDirect = historyDataDirect
        self.testDataDirect = testDataDirect
        self.classList = classList

    def loadTxtFile(self, fileName):
        # load a single file containning the context of an email
        fileData = open(fileName,encoding='latin-1').read()
        regEx = re.compile('\\W+')
        wordList = regEx.split(fileData)
        # print(wordList)
        return wordList

    def loadHistoryData(self):
        # collect the history data as training data.
        classVector = []
        pClass = []
        historyData = []
        for i in range(len(self.classList)):
            direct = self.historyDataDirect + self.classList[i]
            fileNameList = os.listdir(direct)
            pClass.append(len(fileNameList))
            for name in fileNameList:
                data = self.loadTxtFile(direct+'/'+name)
                historyData.append(data)
                classVector.append(i)
        pClass = np.array(pClass)/sum(pClass)
        return historyData, classVector, pClass

    def loadTestingData(self,testingDirect,classList):
        # collect testing data that are concentrated in a single .txt file
        fileName = os.listdir(testingDirect)
        wordList = self.loadTxtFile(testingDirect+'/'+fileName[0])
        classListOfTestingData = []
        sentence = []
        testingData = []
        for word in wordList:
            if word in classList:
                if sentence:
                    # print(sentence)
                    testingData.append(testingData)
                    classListOfTestingData.append(classList.index(word))
                    sentence = []
                else:
                    continue
            else:
                sentence.append(word)
        return testingData, classListOfTestingData

    def createWordsset(self,historyData):
        # creat the word set for history data
        wordSet = set([])
        for d in historyData:
            wordSet = wordSet|set(d)
        return list(wordSet)

    def transferSentence2vector(self, inputSentence, wordSet):
        '''trasfer a sentence (from history or testing data) into a vector to match
        with the word set'''
        vector = [0]*len(wordSet)
        for word in inputSentence:
            if word in wordSet:
                vector[wordSet.index(word)] = 1
            # else:
                # print('The word %s is not in history data.'%word)
        return vector

    def calConditionalProb(self, wordSet, dataVector, classHistory):
        # calculate the P(w_i|A), the conditional probability
        pWord = np.ones((len(wordSet),len(self.classList)))
        numCount = np.ones(len(self.classList))*2
        for i in range(len(classHistory)):
            pWord[:,classHistory[i]] += np.array(dataVector[i])

        countWordperClass = np.sum(pWord,axis=0)
        pWord /= countWordperClass
        return pWord

    def NBclassifer(self, inputVector, pWord, pClass):
        # Clasify a testing data
        pNB = []
        for i in range(len(pClass)):
            p = np.sum(inputVector*np.log(pWord[:,i])+np.log(pClass[i]))
            pNB.append(p)

        return pNB.index(max(pNB))

    def error_rate(self, resultClass, testingClass):
        sizeTesting = len(resultClass)
        errorCount = 0
        for i in range(sizeTesting):
            if resultClass[i] != testingClass[i]:
                errorCount+=1
        return errorCount/sizeTesting

    def run(self):
        historyData, classHistory, pClass = self.loadHistoryData()
        wordSet = self.createWordsset(historyData)
        testingData, classTesting = self.loadTestingData(self.testDataDirect,self.classList)
        historyDataVec = []
        for sent in historyData:
            historyDataVec.append(self.transferSentence2vector(sent, wordSet))
        pWord = self.calConditionalProb(wordSet, historyDataVec, classHistory)

        testingResult = []
        for i, sent in enumerate(testingData):
            print(np.round(i/len(testingData)*100,2),'%',end='\r')
            testingVec = self.transferSentence2vector(sent, wordSet)
            testingResult.append(self.NBclassifer(testingVec, pWord, pClass))

        errorRate = self.error_rate(testingResult, classTesting)
        print(errorRate)

if __name__ == "__main__":

    directHistory = './email_history/'
    directTestingData = './email_testing/'
    classList = ['ham','spam']

    NB_test = NBspamClassifier(directHistory,directTestingData,classList)
    NB_test.run()












































