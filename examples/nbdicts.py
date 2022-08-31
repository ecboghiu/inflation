import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
import numpy as np
import timeit
import sys
import numba

# First create a dictionary using Dict.empty()
# Specify the data types for both key and value pairs

# Dict with key as strings and values of type float array
dict_param1 = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

# Dict with keys as string and values of type float
dict_param2 = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)

# Type-expressions are currently not supported inside jit functions.
float_array = types.float64[:]

@njit
def add_values(d_param1, d_param2):
    # Make a result dictionary to store results
    # Dict with keys as string and values of type float array
    result_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    
    for key in d_param1.keys():
      result_dict[key] = d_param1[key] + d_param2[key]
    
    return result_dict

dict_param1["hello"]  = np.asarray([1.5, 2.5, 3.5], dtype='f8')
dict_param1["world"]  = np.asarray([10.5, 20.5, 30.5], dtype='f8')

dict_param2["hello"]  = 1.5
dict_param2["world"]  = 10

final_dict = add_values(dict_param1, dict_param2)

infl =  [3, 3, 2]    
ins = [2, 3, 2]
outs = [2, 2, 3]
nrparties = len(ins)

PROPERTIES = 6
LETTERS = 12
np.random.seed(123)
m = np.random.randint(3,size=(LETTERS, PROPERTIES), dtype=np.uint8)

dims = np.array((nrparties, *infl, max(ins), max(outs)))

@numba.njit()
def ha(m):
    return m.sum()

@numba.njit()
def nb_operator_hash(op,dims):
    return np.ravel_multi_index(op, dims)

@numba.njit()
def nb_operator_unhash(index,dims):
    return np.unravel_index(index, dims)

@numba.njit()
def nb_ravel_multi_index(op, dims):
    summ = op[0]
    prod = 1
    for i in range(1, op.shape[0]):
        prod *= dims[i-1]
        summ += op[i]*prod
    return summ

@numba.njit()
def nb_hash_op(op: np.ndarray, NB_RADIX: int) -> int:
    summ = op[0]
    prod = 1
    for i in range(1, op.shape[0]):
        prod *= NB_RADIX
        summ += op[i]*prod
    return summ

NB_RADIX = 5
NB_RADIX_VEC = np.array([NB_RADIX**i for i in range(m.shape[0])], dtype=int)
m = m.astype(int)

@numba.njit()
def nb_hash_op2(op: np.array, NB_RADIX_VEC: np.array) -> int:
    s = 0
    for i in range(op.shape[0]):
        s = s + op[i] * NB_RADIX_VEC[i]
    return s
    #return sum(np.dot(op, NB_RADIX_VEC).astype(int))