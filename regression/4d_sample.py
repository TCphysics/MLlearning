import numpy as np
import matplotlib.pyplot as plt
import random
import lwlr

k = 0

def fourDimensionalFunction(x):
    return x[0] + 5*np.sin(x[1])*x[2]+ np.random.normal(0,0.8,size=None)

Num = 200
x = [i/10 for i in range(Num)]
y = [np.sqrt(i) for i in range(Num)]
z = [np.log(i/100+1) for i in range(Num)]
x0 = [[x[i], y[i], z[i]] for i in range(Num)]

w0 = [fourDimensionalFunction(xi) for xi in x0]

wp = []
for xi in x0:
    LWLR = lwlr.localWeightedLinearRegression(x0, w0, k = k, xTest = xi)
    LWLR.run()
    W = LWLR.W
    # print('W:',W[2])
    wi = sum([xi[i] * W[i+1] for i in range(3)])+W[0]
    # print(yi[0])
    wp.append(wi[0])

# quit()
axis = [i for i in range(Num)]
plt.figure(figsize=(8,5))
plt.plot(axis,w0, 'ro',ms=2.5,label='original data')
plt.plot(axis,wp,'b' ,ms=1.5,label='LWLR fitting, k='+str(k))
plt.xlabel('No.(x,y,z)')
plt.legend()
plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.1)
# plt.show()
plt.savefig('4d_lwlr_fitting_k='+str(k)+'.pdf')