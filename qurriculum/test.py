#%%
import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('D:\\Documents\\1. GitHub\\qurriculum_learning\\partially_corrupted_labels\\test_set_1000examples.npy')

print(img_array.shape)
# plt.imshow(img_array, cmap='gray')
# plt.show()

#%%

for i in range(1, 8):
    print((i-1)%8, i, (i+1)%8)


#%%
import numpy as np

nqubits = 32
layers = int(np.log2(nqubits))
qubits = list(range(nqubits))


for j in range(layers-1):
    
    print(qubits)
    len_qubits = len(qubits)
    
    qub = []
    for i in range(len_qubits):
        if i%2 == 1:
            qub.append(qubits[i])
            
    qubits = qub