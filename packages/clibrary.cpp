#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#include <string.h>
#include <cmath>

#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic error "-fpermissive"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
faster_numpy_sum(PyObject *self, PyObject *args)
{
   PyArrayObject *arr1;
   npy_intp *dim;
   PyObject *value;
    
   if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) return NULL;

   dim = PyArray_DIMS(arr1);
   int size = dim[0];
   double sum = 0;
   double* data = (double*) PyArray_GETPTR1(arr1, 0);
   for(int x=0;x<size;x++){
      sum += data[x];
   }
   value = Py_BuildValue("d", sum);
   
   return value;
}

static PyObject *
faster_numpy_mean(PyObject *self, PyObject *args)
{
   PyArrayObject *arr1;
   npy_intp *dim;
   PyObject *value;
    
   if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) return NULL;

   dim = PyArray_DIMS(arr1);
   int size = dim[0];
   double sum = 0;
   double* data = (double*) PyArray_GETPTR1(arr1, 0);
   for(int x=0;x<size;x++){
      sum += data[x];
   }
   value = Py_BuildValue("d", sum/size);
   
   return value;
}

static PyObject *
faster_numpy_std(PyObject *self, PyObject *args)
{
   PyArrayObject *arr1;
   npy_intp *dim;
   PyObject *value;
    
   if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr1)) return NULL;

   dim = PyArray_DIMS(arr1);
   int size = dim[0];
   double sum = 0;
   double* data = (double*) PyArray_GETPTR1(arr1, 0);
   for(int x=0;x<size;x++){
      sum += data[x];
   }
   double mean = sum/size;
   double dev = 0;
   for(int x=0;x<size;x++){
      dev += pow(data[x] - mean, 2);
   }
   dev /= size;
   dev = sqrt(dev);
   value = Py_BuildValue("d", dev);
   
   return value;
}

static PyObject *
faster_numpy_shift(PyObject *self, PyObject *args)
{
   PyArrayObject *arr1;
   npy_intp *dim;
   int value;
    
   if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &arr1, &value)) return NULL;

   dim = PyArray_DIMS(arr1);
   int size = dim[0];
   double* data = (double*)PyArray_GETPTR1(arr1, 0);
   if(value >= 0){
      memmove(&data[0], &data[value], (size-value) * sizeof(double));
   }else{
      memmove(&data[-value], &data[0], (size+value) * sizeof(double));
   }
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject *
faster_numpy_variance(PyObject *self, PyObject *args)
{
   PyArrayObject *npy_actual, *npy_target;
   npy_intp *dim;
   PyObject *py_variance;
    
   if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &npy_actual, &PyArray_Type, &npy_target)) return NULL;

   dim = PyArray_DIMS(npy_actual);
   int size = dim[0];
   double* data1 = (double*)PyArray_GETPTR1(npy_actual, 0);
   double* data2 = (double*)PyArray_GETPTR1(npy_target, 0);
   double sum = 0;
   double *variance;
   py_variance = PyArray_FromDims(1, (int*)dim,  NPY_DOUBLE);
   variance = (double*)PyArray_GETPTR1((PyArrayObject*)py_variance, 0);
   for(int i=0;i<size;i++){
        variance[i] += pow(data1[i] - data2[i], 2);
   }
   return py_variance;
}

static PyMethodDef module_methods[] = {
	{"sum", (PyCFunction)faster_numpy_sum, METH_VARARGS,
	 "sum"
   },
	{"mean", (PyCFunction)faster_numpy_mean, METH_VARARGS,
	 "average"
   },
	{"std", (PyCFunction)faster_numpy_std, METH_VARARGS,
	 "standard deviation"
   },
	{"shift", (PyCFunction)faster_numpy_shift, METH_VARARGS,
	 "shift"
   },
	{"variance", (PyCFunction)faster_numpy_variance, METH_VARARGS,
	 "variance"
   },
	{NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "clibrary",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};


#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_clibrary(void)
#else
#define INITERROR return

PyMODINIT_FUNC
initclibrary(void)
#endif
{
    PyObject* m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("clibrary", module_methods, "");
#endif
    
    import_array();


    if (m == NULL)
#if PY_MAJOR_VERSION >= 3
        return m;
#else
        return;
#endif
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
