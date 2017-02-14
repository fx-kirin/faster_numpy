
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "structmember.h"
#include "numpy/arrayobject.h"
#include <string.h>
#include <cmath>

#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic error "-fpermissive"

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
   for(int x;x<size;x++){
      double* data = PyArray_GETPTR1(arr1, x);
      sum += *data;
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
   for(int x;x<size;x++){
      double* data = PyArray_GETPTR1(arr1, x);
      sum += *data;
   }
   value = Py_BuildValue("d", sum/size);
   return value;
}

static PyMethodDef module_methods[] = {
	{"mean", (PyCFunction)faster_numpy_mean, METH_VARARGS,
	 "average"
   },
	{NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initclibrary(void)
{
    PyObject* m;

    m = Py_InitModule3("clibrary", module_methods,
                       "");
    import_array();

    if (m == NULL)
      return;
}
