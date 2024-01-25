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



#%%
import numpy as np

cl_ratios =  [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cv_size_batches, cv_limits_it = [], []
limit_iter, size_batch = 0, 0

labels_train = [0,0,0,1,0,2,1,2,0,1,0,2,1,1,1,0,2,2,1,0]

num_iters = 1000
it = 999

for i, ratio in enumerate(cl_ratios):
    
    if i < len(cl_ratios)-1:
        limit_iter += round(ratio*num_iters)
        size_batch += round(ratio*len(labels_train))
    else:
        limit_iter = num_iters
        size_batch = len(labels_train)
    
    cv_limits_it.append(limit_iter)
    cv_size_batches.append(size_batch)

index_size_batch = np.argmax(it < np.array(cv_limits_it)) # This gives you the first occurrence where the condition is met
cv_size_batch = cv_size_batches[index_size_batch]

labels_train_batch = labels_train[:cv_size_batch]

print(labels_train_batch)

# %%

cl_ratios = [0.4, 0.3, 0.2, 0.1]

num_iters = 1000
labels_train = [0,0,0,1,0,2,1,2,0,1,0,2,1,1,1,0,2,2,1,0]

size_batch, limit_iter = 0, 0
cv_limits_it, cv_size_batches = [], []

for i, ratio in enumerate(cl_ratios):
                
    if i < len(cl_ratios)-1:
        limit_iter += round(1/len(cl_ratios)*num_iters) #round(ratio*num_iters)
        size_batch += round(ratio*len(labels_train))
    else:
        limit_iter = num_iters
        size_batch = len(labels_train)
    
    cv_limits_it.append(limit_iter)
    cv_size_batches.append(size_batch)

print(cv_limits_it)
print(cv_size_batches)


#%%
import numpy as np

def norm(a,b,c,d):
    return a**2 + b**2 + c**2 + d**2

def mse(a,b,c,d):
    return (1-a)**2 + b**2 + c**2 + d**2

a = 1.9
b = 2.87
c = 3.27
d = 5.28

p = np.array([a,b,c,d])

print(mse(a,b,c,d))
print(1+np.linalg.norm(p)**2-2*a)

# %%
from shapely.geometry import Polygon, Point
import numpy as np

num_points = 10**6

# Define coordinates of the points of each region# Definir las coordenadas de los puntos de cada región
region01_coords = np.array([(-2, 1), (2, 1), (4, 3), (4, 4), (-4, 4), (-4, 3)])    # Class 0
region02_coords = np.array([(-3, -4), (0, -1), (3, -4)])                           # Class 0
region1_coords = np.array([(0, -1), (3, -4), (4, -4), (4, 3)])                     # Class 1
region2_coords = np.array([(0, -1), (-3, -4), (-4, -4), (-4, 3)])                  # Class 2
region3_coords = np.array([(-2, 1), (2, 1), (0, -1)])                              # Class 3

def labeling(x, y):

    # Create Polygons for each region
    region01_poly = Polygon(region01_coords)
    region02_poly = Polygon(region02_coords)
    region1_poly = Polygon(region1_coords)
    region2_poly = Polygon(region2_coords)
    region3_poly = Polygon(region3_coords)
    
    p = Point(x, y)
    if region01_poly.contains(p):
        return 0
    elif region02_poly.contains(p):
        return 0
    elif region1_poly.contains(p):
        return 1
    elif region2_poly.contains(p):
        return 2
    elif region3_poly.contains(p):
        return 3
    else:
        return None # if the point is not in any region
    
j_list = np.random.uniform(-4, 4, (num_points,2))

num_points_3 = 0
for j in j_list:
    if labeling(*j) == 3:
        num_points_3 += 1

print(round((num_points-num_points_3)/num_points*100, 2), "%")

#%%

import jax.scipy.optimize
import jax.numpy as jnp

def func(x, y, b, c):
    return x**2 + y**2 + b - c

def func2(x, b, c):
    return func(x[0], x[1], b, c)


b = 10
c = -10

# print(fun(, *args))

a = jax.scipy.optimize.minimize(func2, jnp.array([200.0, 300.0]), args=(b,c), method="BFGS", options={"maxiter":1})

print(a.x.tolist())
print(a.fun.tolist())
print(a.success)
print(a.nit)

#%%

import jax.lax
import jax.numpy as jnp
import numpy as np

gs_train = jnp.array([0,1,2,3,4,5,6,7,8,9])
cl_size_batch = 3

start = jnp.array([0])
size = jnp.array([cl_size_batch])

gs_train_batch = jax.lax.dynamic_slice(gs_train, start, size)

print(gs_train_batch)