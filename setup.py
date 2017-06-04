from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension('faster_numpy.cylib', sources=['faster_numpy/cylib.pyx'], include_dirs=[numpy.get_include()]),
    Extension('faster_numpy.clibrary', sources = ['packages/clibrary.cpp'], include_dirs=[numpy.get_include()])
]

setup(
    name='faster_numpy',
    packages=['faster_numpy'],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
