import operator
from os import listdir
from random import shuffle
import numpy as np
from network_lib import netural_network


def single_2_by_2(L):
    Ln = np.array(L).reshape(4,4)
    count = 0
    for i in range(3):
        for j in range(3):
            s = Ln[i,j]+Ln[i+1,j]+Ln[i,j+1]+Ln[i+1,j+1]
            count = count +1 if s == 4 else count

    if count == 1:
        return True
    else:
        return False

def four_by_four_data(inputSize, dataSize):
    inData, outData = [], []
    # count = 0
    for i in range(inputSize):
        temp = [np.random.randint(2) for j in range(dataSize-2)]+[0]*2
        out = [1,0] if single_2_by_2(temp) else [0,1]
        # count = count+1 if out == [1,0] else count
        inData.append(temp)
        outData.append(out)
    # print(count/inputSize)
    return inData, outData

def test(inData, outData, W, B):
    Error = 0
    for i in range(len(inData)):
        NNtest = netural_network(NNlayers, inData[i], outData[i], W, B)
        NNtest.forward_propagation()
        result = [1,0] if NNtest.A[-1][0] > NNtest.A[-1][1] else [0,1]
        # print(result, outData[i])
        if result != outData[i]:
            Error += 1
        del(NNtest)
    return Error/len(inData)


inputSize, dataSize, Niteration = 500, 16, 10000
inData, outData = four_by_four_data(inputSize, dataSize)
# quit()
############ NN running #################################
NNlayers = []
Sizes = [dataSize] + NNlayers + [len(outData[0])]
Weights = [np.random.random((y,x)) for x,y in zip(Sizes[:-1], Sizes[1:])]
Bias = [np.random.random((x,1)) for x in Sizes[1:]]

print('Netural work program starts, network layers is',NNlayers)
i, count, cmax = 0, 0, 0
for j in range(Niteration*inputSize):
    NN = netural_network(NNlayers, inData[i], outData[i], Weights, Bias)
    NN.run()
    count = count+1 if NN.error<1e-2 else 0
    cmax = max(cmax, count)
    if count > inputSize:
        break
    per = j/Niteration/inputSize*1e2
    print('error:','%.4f, cmax: %d, count: %d, running: %.1f'%(NN.error,cmax,count,per),'%  ',end='\r')
    Weights, Bias = NN.Weights, NN.Bias
    i = (i+1)%inputSize
    del(NN)

print('Netural work program ends!')
print('Training error rate:')

############ Training result ################################
print(test(inData, outData, Weights, Bias)*100,'%')
# quit()

############ Testing result #################################

testInSize, dataSize = 500, 16
testInData, testOutData = four_by_four_data(testInSize, dataSize)

print('Testing error rate:')
print(test(testInData, testOutData, Weights, Bias)*100,'%')
quit()








