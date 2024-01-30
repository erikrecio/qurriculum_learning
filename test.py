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

#%%
import jax
import jax.numpy as jnp

@jax.jit
def f(arr, condition):
    return jnp.where(condition, arr, 0)

a = jnp.array([1,2,3,4,5])
condition = jnp.array([True,False,True,False,True])

print(a[condition])
print(f(a, condition))

#%%
import jax.numpy as jnp
from functools import partial

labels = jnp.array([-1,0,0,1,0,2,1,2,0,1,0,-1,1,1,1,0,2,2,1,0])


print(labels[labels!=-1])
print(len(labels))
print(jnp.where(labels==-1, 0, 1).sum())
print(1/(labels==-1).sum())
cl_size_batch = 10
condition_labels = jnp.concatenate((jnp.full((cl_size_batch,), True), jnp.full((len(labels)-cl_size_batch,), False)))
print(condition_labels)



@partial(jax.jit, static_argnames=["label"])
def single_loss(q, r, s, t, label):
    cost = 0
    if label != -1:
        s = r[label]
        cost = label
        
    
    return cost
for i in range(-2, 5):
    print(single_loss(3,[4,5,6,7,8,9,0],5,6,i))
    


#%%

def f(x):
    global b, a
    a = 4
    b = 2
    return x+a

a = 3
print(f(0))

print(a, b)

#%%

def f(x):
    
    def g(a):
        # nonlocal x
        x = 2
        return a+x
    a = g(1)
    print(x)
    return a

print(f(3))

#%%
import jax
import jax.numpy as jnp
import jaxopt
import scipy.optimize
jax.config.update("jax_enable_x64", True)

@jax.jit
def f(x): #,a
    return x**2.0 # + a + jnp.exp(x)



init_params = 2.0
# a = 2
# args = [a]
stepsize = 0.01
maxiter = 50

# opt = jaxopt.GradientDescent(f, stepsize=stepsize, jit=True)

opt = jaxopt.BFGS(f, stepsize=stepsize, jit=True, maxiter=maxiter)
# opt = jaxopt.LBFGS(f, stepsize=stepsize, jit=True, maxiter=maxiter)

# opt = jaxopt.ScipyMinimize(method="BFGS", fun=f, maxiter=maxiter)

# params = init_params
# state = opt.init_state(init_params, *args)
# print("State - ", state)

# for i in range(maxiter):
#     output = opt.update(params, state, *args)
#     print(output)
#     params, state = output

result = opt.run(init_params)#, *args)
print(result)

a, b = result

print(a)
print(b.value) #value or fun_val


scipy_res = scipy.optimize.minimize(f, init_params, method="BFGS", options={"maxiter":maxiter}) #, args=(a)
print(scipy_res)

#%%
import jax
import jax.numpy as jnp
import jaxopt
import scipy.optimize
jax.config.update("jax_enable_x64", True)

def f(x):
    return x**2 + jnp.exp(x)

maxiter = 5
init_params = 2.0

opt = jaxopt.BFGS(fun=f, maxiter=maxiter)
x, state = opt.run(init_params)
print()
print(f"x = {x}")
print(f"f(x) = {state.value}")

print()

opt2 = jaxopt.ScipyMinimize(fun=f, method="BFGS", maxiter=maxiter)
x2, state2 = opt2.run(init_params)
print(f"x2 = {x2}")
print(f"f(x2) = {state2.fun_val}")

#%%
import jax
import jax.numpy as jnp
import jaxopt
import optax
import scipy.optimize
jax.config.update("jax_enable_x64", True)
from time import time as t
import os
import contextlib

def f(x):
    return x**2 + jnp.exp(x)

def my_run(params, state):
    for i in range(maxiter):
        params, state = opt.update(params, state)
    return params, state

def my_run2(opt, params, state, maxiter):
    it = 0
    while it <= maxiter - 1:
        val = opt.update(params, state)
        it += 1
    return val

maxiter = 3000
init_params = 200.0

x = init_params
# print(x**2 + jnp.exp(x))

# opt = jaxopt.BFGS(f, verbose = False, maxiter = maxiter, jit = True)
opt = jaxopt.OptaxSolver(f, optax.adam(0.01), verbose=False, jit=True, maxiter=maxiter)

params = init_params
state = opt.init_state(init_params)
start = t()
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    params, state = my_run2(opt, params, state, maxiter)
print("loop2 - ", t()-start)
print(state)

params = init_params
state = opt.init_state(init_params)
start = t()
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    params, state = opt.run(params)
print("run - ", t()-start)
print(state)

params = init_params
state = opt.init_state(init_params)
start = t()
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    params, state = my_run(params, state)
print("loop - ", t()-start)
print(state)






#%%

import pennylane as qml

class AdamOptimizer2(qml.AdamOptimizer):
    def apply_grad(self, grad, args):
        r"""Update the variables args to take a single optimization step. Flattens and unflattens
        the inputs to maintain nested iterables as the parameters of the optimization.

        Args:
            grad (tuple[ndarray]): the gradient of the objective
                function at point :math:`x^{(t)}`: :math:`\nabla f(x^{(t)})`
            args (tuple): the current value of the variables :math:`x^{(t)}`

        Returns:
            list: the new values :math:`x^{(t+1)}`
        """
        args_new = list(args)

        if self.accumulation is None:
            self.accumulation = {"fm": [0] * len(args), "sm": [0] * len(args), "t": 0}

        self.accumulation["t"] += 1

        # Update step size (instead of correcting for bias)
        new_stepsize = (
            self.stepsize
            * pnp.sqrt(1 - self.beta2 ** self.accumulation["t"])
            / (1 - self.beta1 ** self.accumulation["t"])
        )

        trained_index = 0
        for index, arg in enumerate(args):
            self._update_accumulation(index, grad[trained_index])
            args_new[index] = arg - new_stepsize * self.accumulation["fm"][index] / (
                pnp.sqrt(self.accumulation["sm"][index]) + self.eps
            )
            trained_index += 1

        return args_new

import jax
import jax.numpy as jnp
import pennylane.numpy as pnp
import jaxopt
import optax
import numpy as np
from functools import partial

import os
import contextlib

from time import time as t
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

num_iters = 3000
cl_types = ["NCL", "CL", "ACL"]#, "CL", "ACL", "SPCL", "SPACL"]
cl_batch_ratios = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cl_iter_ratios  = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
batch_size = 20

@jax.jit
def single_cost(w, index):
    return gs_train[index]*w**2 + labels_train[index]*jnp.exp(w)

@jax.jit
def cost(w, data):
    result = jax.vmap(single_cost, in_axes=[None, 0])(w, data)
    return result.sum()

# @jax.jit
def grad_cost(w, data):
    grad_w = jax.grad(cost, argnums=0)(w, data)
    return grad_w, 0

# @partial(jax.jit, static_argnames={"opt"})
# def optimizing_step(opt, params, state, batch_range):
        
#     params, state = opt.update(params, state, batch_range)
    
#     return params, state
    
def data_iterator():
    for it in range(num_iters):
        yield batch_range_arr[it]

# @partial(jax.jit, static_argnames="cl")
def train(cl):
    
    global batch_range_arr
    batch_range_arr = []
    
    start = t()
    if cl in ["CL", "ACL", "SPCL", "SPACL"]:
        for it in range(num_iters):
            cl_size_batches, cl_limits_it = [], []
            limit_iter, size_batch = 0, 0
            
            steps = len(cl_iter_ratios)
            for i in range(steps):
                
                if i < steps-1:
                    limit_iter += round(cl_iter_ratios[i]*num_iters)
                    size_batch += round(cl_batch_ratios[i]*len(labels_train))
                else:
                    limit_iter = num_iters
                    size_batch = len(labels_train)
                
                cl_limits_it.append(limit_iter)
                cl_size_batches.append(size_batch)
            
            index_size_batch = np.argmax(it < np.array(cl_limits_it)) # This gives you the first occurrence where the condition is met
            cl_size_batch = jnp.array(cl_size_batches)[index_size_batch]
            range_test = range(cl_size_batch)
            batch_range = jnp.array(list(range_test))
            batch_range_arr.append(batch_range)
            
    elif cl == "NCL":
        for it in range(num_iters):
            batch_range = jnp.array(list(range(20)))
            batch_range_arr.append(batch_range)
    print("calc", " - ", t() - start)
    # for it in range(num_iters):
    #     print(batch_range_arr[it])
    
    w_init = 10.0
    b_init = 10.0
    opt = jaxopt.BFGS(cost, verbose=False, jit=True)
    
    start = t()
    if True:
        # opt = jaxopt.OptaxSolver(cost, optax.adam(0.01), verbose=False, jit=True, maxiter=num_iters)
        
        params = init_params
        state = opt.init_state(init_params, jnp.array(list(range(20))))
        iterator = data_iterator()
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            for it in range(num_iters):
                params, state = opt.update(params, state, next(iterator))
        
        print("value = ", state.value)
        
        # resultadu = opt.run_iterator(params, iterator)
        # print(resultadu)
    else:
        opt = AdamOptimizer2(stepsize=0.01)
        
        params = init_params
        
        for it in range(num_iters):
            ([params, _], l) = opt.step_and_cost(cost, params, batch_range_arr[it], grad_fn=grad_cost)
            
            
    print(cl, " - ", t() - start)
    
    
def main():
    
    for run in range(5):
    
        global gs_train, labels_train
        gs_train = jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        labels_train = jnp.array([1,2,1,2,2,0,0,2,0,1,0,2,1,0,2,1,0,2,2,2])
        
        
        for cl in cl_types:
            train(cl)
            
    
    
main()


#%%

data = np.empty(20, dtype=object)
for i in range(data.shape[0]):
    data[i] = list(range(i))
print(data)

#%%
import os
import contextlib

with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    print("This won't be printed.")
print("hello")

#%%
num_iters = 50
train_size = 100
cl_batch_ratios = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cl_iter_ratios  = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]

cl_batches = []
i_batch_size = 0

for i in range(len(cl_iter_ratios)):
    if i < len(cl_iter_ratios)-1:
        i_batch_size += int(cl_batch_ratios[i]*train_size)
        i_num_iters = int(cl_iter_ratios[i]*num_iters)
    else:
        i_batch_size = train_size
        i_num_iters = num_iters - len(cl_batches)
        
    cl_batches += [i_batch_size]*i_num_iters
    
print(cl_batches)


#%%


import jaxopt
import optax
from time import time
import jax.numpy as jnp
import jax


# @jax.jit
def cost(w, data):
    result = 0
    for d in data:
        result += d*w**2 + jnp.exp(w)
    return result

data = [1,2,3,4,5,6,7,8,9,10]
batches = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10]
num_iters = len(batches)
init_params = 200.0
# opt = jaxopt.OptaxSolver(cost, optax.adam(0.01), verbose=False, jit=True)


# --------------------------------------------------------------- #
# ---------------------- Not using batches ---------------------- #
# --------------------------------------------------------------- #
start = time()
params = init_params
state = opt.init_state(init_params, data)

for it in range(num_iters):
    params, state = opt.update(params, state, data)

print("Fixed data time - ", round(time() - start, 2), "s")


# --------------------------------------------------------------- #
# ------------------------ Using batches ------------------------ #
# --------------------------------------------------------------- #
start = time()
params = init_params
state = opt.init_state(init_params, data)

for it in range(num_iters):
    params, state = opt.update(params, state, data[:batches[it]])

print("Iterator time - ", round(time() - start, 2), "s")



#%%

from time import time
import jax.numpy as jnp
import jax

# @jax.jit
# def cost(data):
#     return sum([jnp.exp(i) for i in range(1000)])

data = jnp.array([1,2,3,4,5,6,7,8,9,10])

start = time()
cost(data[:7])
print("Time -", round((time() - start)*1000, 2), "ms")


#%%

import jax.numpy as jnp

ascending = True

a = jnp.array([0,2,9,8,1,3,7,8,0,10])
b = jnp.array([10,9,8,7,6,5,4,3,2,1])

p = jnp.where(ascending, a.argsort(), a.argsort()[::-1])

print(p)

print(a[p])
print(b[p])
# print(c[::-1])

#%%

import pennylane as qml
import jax
import jax.numpy as jnp
import numpy as np
from time import time

nqubits = 8
num_points = 1100

def X(i):
    return qml.PauliX(i)

def Z(i):
    return qml.PauliZ(i)

@jax.jit
def ground_state(j1, j2):
    
    hamiltonian = 0
    for i in range(nqubits):
        hamiltonian += Z(i)
        hamiltonian -= j1 * X(i) @ X((i+1)%nqubits)
        hamiltonian -= j2 * X((i-1)%nqubits) @ Z(i) @ X((i+1)%nqubits)
    
    ham_matrix = qml.matrix(hamiltonian)
    _, eigvecs = jnp.linalg.eigh(ham_matrix)
    
    return eigvecs[:,0]


j_list = np.random.uniform(-4, 4, (num_points,2))

start = time()
gs_list = []
for i in range(num_points):
    gs_list.append(ground_state(j_list[i,0], j_list[i,1]))
    
# gs_list = jax.vmap(ground_state, in_axes=[0,0])(j_list[:,0], j_list[:,1])
print("t - ", time()-start)
print(len(gs_list))

#%%
from shapely.geometry import Polygon, Point
import shapely
import jax
import matplotlib.path

class Point2(Point):
    def __new__(self, *args):
        if len(args) == 0:
            # empty geometry
            # TODO better constructor
            return shapely.from_wkt("POINT EMPTY")
        elif len(args) > 3:
            raise TypeError(f"Point() takes at most 3 arguments ({len(args)} given)")
        elif len(args) == 1:
            coords = args[0]
            if isinstance(coords, Point):
                return coords

            # Accept either (x, y) or [(x, y)]
            if not hasattr(coords, "__getitem__"):  # generators
                coords = list(coords)
            coords = np.asarray(coords).squeeze()
        else:
            # 2 or 3 args
            coords = jnp.array(args).squeeze()

        if coords.ndim > 1:
            raise ValueError(
                f"Point() takes only scalar or 1-size vector arguments, got {args}"
            )
        if not np.issubdtype(coords.dtype, np.number):
            coords = [float(c) for c in coords]
        geom = shapely.points(coords)
        if not isinstance(geom, Point):
            raise ValueError("Invalid values passed to Point constructor")
        return geom
    
# @jax.jit
def pointing(x, y):
    return Point2(x, y)

# pointing(1.0,1.0)

region01_coords = np.array([(-2, 1), (2, 1), (4, 3), (4, 4), (-4, 4), (-4, 3)])    # Class 0
region02_coords = np.array([(-3, -4), (0, -1), (3, -4)])                           # Class 0
region1_coords = np.array([(0, -1), (3, -4), (4, -4), (4, 3)])                     # Class 1
region2_coords = np.array([(0, -1), (-3, -4), (-4, -4), (-4, 3)])                  # Class 2
region3_coords = np.array([(-2, 1), (2, 1), (0, -1)])                              # Class 3

# polygon = region3_coords
# points = np.array([0,0])

# path = matplotlib.path.Path(polygon)
# inside2 = path.contains_point(points)

# string = "hello" if inside2 else "byebye"


def labeling(p):

    # Create Polygons for each region
    region01_poly = matplotlib.path.Path(region01_coords)
    region02_poly = matplotlib.path.Path(region02_coords)
    region1_poly = matplotlib.path.Path(region1_coords)
    region2_poly = matplotlib.path.Path(region2_coords)
    region3_poly = matplotlib.path.Path(region3_coords)
    
    if region01_poly.contains_point(p):
        return 0
    elif region02_poly.contains_point(p):
        return 0
    elif region1_poly.contains_point(p):
        return 1
    elif region2_poly.contains_point(p):
        return 2
    elif region3_poly.contains_point(p):
        return 3
    else:
        return None # if the point is not in any region


def labeling2(x, y):

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
    
num_points = 10000

j_list = np.random.uniform(-4, 4, (num_points,2))

start=time()
for i in range(num_points):
    label1 = labeling(j_list[i])
print("label1 t -", time()-start)

start=time()
for i in range(num_points):
    label2 = labeling2(j_list[i,0], j_list[i,1])
print("label2 t -", time()-start)




# print(string)