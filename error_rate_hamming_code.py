import numpy as np
import matplotlib.pyplot as plt
import time as time

################################################################### create message bits
###################################################################
tic = time.time()

##Generating random message bits
n = 50000
n2 = 50050
m = np.random.randint(low=0,high=2, size=n).astype('int')
m2 = np.random.randint(low=0,high=2, size=n2).astype('int')
hammingcode74 = []
hammingcode1511 = []
m = m.reshape(-1,4)
m2 = m2.reshape(-1,11

)
generator1 = np.matrix([[0,1,1,1,0,0,0],[1,0,1,0,1,0,0],[1,1,0,0,0,1,0],[1,1,1,0,0,0,1]]) ## Generator Matrix for 7,4


generator2 = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], ## Generator Matrix for 15,11
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])


H1 = np.matrix([[1,0,0,0,1,1,1],[0,1,0,1,0,1,1],[0,0,1,1,1,0,1]]) ## Parity Check matrix for 7,4

H2 = np.matrix([[1,0,0,0,1,0,0,1,1,0,1,0,1,1,1], # Parity Check for 15,11
                [0,1,0,0,1,1,0,1,0,1,1,1,1,0,0],
                [0,0,1,0,0,1,1,0,1,0,1,1,1,1,0],
                [0,0,0,1,0,0,1,1,0,1,0,1,1,1,1] ])



## Message multiplyied by the codeword
for i in m:
    hold1 = i
    hold2 = ((hold1*generator1)%2)
    hammingcode74.append(hold2)

for i in m2:
    hold1 = i
    hold2 = ((hold1*generator2)%2)
    hammingcode1511.append(hold2)


## converting from list of matrixes to just matrix
listToMatrix = np.asarray(hammingcode74, dtype=int)
listToMatrix = listToMatrix.reshape(len(hammingcode74), -1)

listToMatrix2 = np.asarray(hammingcode1511, dtype=int)
listToMatrix2 = listToMatrix2.reshape(len(hammingcode1511), -1)


kittycHat = []
kittycHat2 = []
epsilon = np.arange(-3, -.29, 0.01)

##BSC error implementation
for p in epsilon:
    epsilonholder = 10**(p)
    errorBits = np.random.rand(listToMatrix.shape[0],listToMatrix.shape[1]) < epsilonholder
    cHat = np.logical_xor(errorBits,listToMatrix).astype('int')
    kittycHat.append(cHat)



for r in epsilon:
    epsilonholder = 10**(r)
    errorBits = np.random.rand(listToMatrix2.shape[0],listToMatrix2.shape[1]) < epsilonholder
    cHat2 = np.logical_xor(errorBits,listToMatrix2).astype('int')
    kittycHat2.append(cHat2)

print(kittycHat)




## Calculating BER1
count =0
BER1 = []
for t in kittycHat:
    #print(count)
    count = count +1

    tempChat = t

    cHatChunk = []
    errorsinChunk = 0
    for x in tempChat:
        x = np.asmatrix(x)
        x = x.transpose()
        temp = ((H1*x)%2)
        temp = np.sum(temp)
        if temp > 0:
            errorsinChunk +=1

    logBER = errorsinChunk/n
    logBER = np.log10(logBER)
    BER1.append(logBER)


BER2 = []
for e in kittycHat2:
    print(count)
    count = count +1

    tempChat = e

    cHatChunk = []
    errorsinChunk = 0
    for x in tempChat:
        x = np.asmatrix(x)
        x = x.transpose()
        temp = ((H2*x)%2)
        temp = np.sum(temp)
        if temp > 0:
            errorsinChunk +=1

    logBER = errorsinChunk/n
    logBER = np.log10(logBER)
    BER2.append(logBER)





toc = time.time()
print("Time to Render", (toc-tic))

## Printing Stuff
plt.plot(epsilon,BER1, label = '(7,4) Hamming Code')
plt.plot(epsilon,BER2, label = "(15,11) Hamming Code")
plt.ylabel('LOG10 RHO')
plt.xlabel('Bit Error Rate')
plt.title('Bit Error Rate V. RHO')
plt.legend()
plt.show()
