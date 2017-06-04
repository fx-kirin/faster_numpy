#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
from libc.string cimport memmove

cpdef void shift(np.ndarray[double, ndim=1, mode="c"] np_array, int value):
    cdef int array_size
    array_size = np_array.shape[0]
    if(value >= 0):
        memmove(&np_array[0], &np_array[value], (array_size-value) *sizeof(double))
    else:
        memmove(&np_array[-value], &np_array[0], (array_size-value) *sizeof(double))
    return

cpdef double mean(np.ndarray[double, ndim=1, mode="c"] np_array):
    cdef int array_size
    cdef double sum_ = 0
    cdef double result
    cdef int x
    array_size = np_array.shape[0]
    for x in range(array_size):
        sum_ += np_array[x]
        
    result = sum_ / array_size
        
    return result

cpdef variance(np.ndarray[double, ndim=1, mode="c"] np_actual, np.ndarray[double, ndim=1, mode="c"] np_target):
    cdef int array_size1, array_size2
    cdef double sum_ = 0
    cdef int x
    array_size1 = np_actual.shape[0]
    cdef np.ndarray[double, ndim=1, mode="c"] value = np.zeros(array_size1)
    for x in range(array_size1):
        value[x] = pow(np_actual[x] - np_target[x], 2)
        
    return value
