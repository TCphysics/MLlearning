import numpy as np

class netural_network(object):

    def __init__(self, neturalSize, inData, outData, Weights, Bias):
        self.inSize = len(inData)
        self.outSize = len(outData)
        self.x = np.array(inData).reshape((self.inSize,1))
        self.y = np.array(outData).reshape((self.outSize,1))
        self.Sizes = [self.inSize] + neturalSize + [self.outSize]
        self.Weights = Weights
        self.Bias = Bias
        self.Z = [np.zeros((x,1)) for  x in self.Sizes[1:]]
        self.A = [np.zeros((x,1)) for  x in self.Sizes]
        self.A[0] = self.stimulationg_func(self.x)
        self.error = 0

    def stimulationg_func(self, z):
        #sigmoid
        return 1 / (1 + np.exp(-z))

    def derivative_func(self, a):
        #sigmoid
        return a * (1-a)

    def delta_calculation(self, w,delta,a):
        return np.dot(w.T, delta) * self.derivative_func(a)

    def forward_propagation(self):
        for i in range(len(self.Sizes)-1):
            self.Z[i] = np.dot(self.Weights[i], self.A[i])+self.Bias[i]
            self.A[i+1] = self.stimulationg_func(self.Z[i])

    def back_propagation(self):
        self.error = np.sum(np.square(self.A[-1] - self.y))
        delta = self.stimulationg_func(self.A[-1]) * (self.A[-1] - self.y) 
        for i in range(len(self.Sizes)-1,0,-1):
            # print(self.A[i-1].shape, delta.shape)
            dW = np.dot(self.A[i-1], delta.T).T
            # print(delta.shape,  self.Bias[i-1].shape)
            self.Bias[i-1] -= 1 * self.error * delta
            delta = self.delta_calculation(self.Weights[i-1], delta, self.A[i-1])
            self.Weights[i-1] -= 1 * self.error * dW
        return


    def run(self):
        self.forward_propagation()
        self.back_propagation()


if __name__ == '__main__':
    def sample_data(inputSize, dataSize):
        inData, outData = [], []
        for i in range(inputSize):
            temp = [np.random.randint(2) for j in range(dataSize)]
            out = [1,0] if 2*sum(temp)>len(temp) else [0,1]
            inData.append(temp)
            outData.append(out)
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


    inputSize, dataSize, Niteration = 100, 10, 5000
    inData, outData = sample_data(inputSize, dataSize)

    NNlayers = []
    Sizes = [dataSize] + NNlayers + [len(outData[0])]
    Weights = [np.random.random((y,x)) for x,y in zip(Sizes[:-1], Sizes[1:])]
    Bias = [np.random.random((x,1)) for x in Sizes[1:]]

    print('Netural work program starts, network layers is',NNlayers)
    i, count, cmax = 0, 0, 0
    for j in range(Niteration*inputSize):
        NN = netural_network(NNlayers, inData[i], outData[i], Weights, Bias)
        NN.run()
        # result = [1,0] if NN.A[-1][0] < NN.A[-1][1] else [0, 1]
        # print(result, outData[i])
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
    print(test(inData, outData, Weights, Bias)*100,'%')
    # quit()

    testInSize, dataSize = 500, 10
    testInData, testOutData = sample_data(testInSize, dataSize)

    print('Testing error rate:')
    print(test(testInData, testOutData, Weights, Bias)*100,'%')
    quit()











