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

#%%
import numpy as np

rng = np.random.RandomState(0)
print("Repeatable test:", [rng.randint(10) for i in range(10)])

print(np.random.randint(10))


#%%
import numpy as np

def alehop():
    return np.array([1,2,3])

a, b, c = alehop() + np.array([1,2,3])
print(a)

z = np.random.randint(0, 10, 10)
print(z)
print(np.argmin(z))

#%%
import jax
import jax.numpy as jnp
import time

def fun(a,b):
    time.sleep(1)
    return a+b

a = jnp.array(1)
b = jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

results = jax.vmap(fun, in_axes=[None, 0])(a, b)

print(results)


# %%

import pandas as pd


results = {}
results["type_cv"] = ['NCV']
results["num_runs"] = [10000]
results["best_run_max"] = [1]
results["best_run_last"] = [1]
results["best_it_max"] = [1]
results["best_it_last"] = [1]
results["best_acc_train_max"] = [1]
results["best_acc_train_last"] = [1]
results["best_acc_val_max"] = [1]
results["best_acc_val_last"] = [1]
results["avg_acc_train_max"] = [1]
results["avg_acc_train_last"] = [1]
results["avg_acc_val_max"] = [1]
results["avg_acc_val_last"] = [1]
results = pd.DataFrame(results)








nqubits = 8
num_iters = 1000
time_now = "2024-01-18 11-33-52"

folder_name = f"Results/{nqubits}q - {num_iters} iters"
results_file_name = f"{folder_name}/{time_now} - Results.csv"




read_results = pd.read_csv(results_file_name)
row_index = read_results.loc[read_results["type_cv"] == "hello"].index

if row_index.shape != (0,):
    read_results.drop(labels=row_index[0], axis=0, inplace=True)
    
results = pd.concat([read_results, results], ignore_index=True)













