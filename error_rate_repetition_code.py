#
#/BER = sum(xor(m*,m))/ (#m)
# BER <= E
#
import numpy as np
import matplotlib.pyplot as plt
import time as time

tic = time.time()
################################################################### create message bits
n = 1000000
m = np.random.randint(low=0,high=2, size=n).astype('bool')
#print("Message Bits\n", m)

################################################################### Create code bits
r = 3
r2 = 5
c = np.tile(m,(r,1)) # takes the matrix and tiles it
c2 = np.tile(m,(r2,1))
#print("Codewords for R = 3\n",c)
#print("Codewords for R = 5\n",c2)

###################################################################  txt through bsc

epsilon = np.arange(-3, -.29, 0.01)
ber_Array = []
ber_Array2 = []

for i in epsilon:
    epsilonholder = 10**(i)
    errorBits = np.random.rand(c.shape[0],c.shape[1]) < epsilonholder
    errorBits2 = np.random.rand(c2.shape[0],c2.shape[1]) < epsilonholder

#print("Error Bits for R = 3\n",errorBits)
    cHat = np.logical_xor(errorBits,c)
#print("cHat for R = 3\n",cHat)
#print("Error Bits for R = 5\n",errorBits2)
    cHat2 = np.logical_xor(errorBits2,c2)
#print("cHat for R = 5\n",cHat2)




###################################################################  decode
    sums = np.sum(cHat, axis=0)
    mHat = sums>r/2
    sums2 = np.sum(cHat2, axis=0)
    mHat2 = sums2>r/2
#print("mHat\n",mHat)

################################################################### compute BER

    BER = np.sum(np.logical_xor(m,mHat))/n
    BER2 = np.sum(np.logical_xor(m,mHat2))/n
    logBER = np.log10(BER)
    logBER2 = np.log10(BER2)
    ber_Array.append(logBER)
    ber_Array2.append(logBER2)
#print("ber\n",ber_Array)
toc = time.time()
print(toc-tic)

################################################################### Plotting
plt.plot(epsilon,ber_Array, label='r = 3')

plt.plot(epsilon,ber_Array2, label='r = 5')

plt.ylabel('LOG10 RHO')
plt.xlabel('Bit Error Rate')
plt.title('Bit Error Rate V. RHO')
plt.legend()
plt.show()


###################################################################
