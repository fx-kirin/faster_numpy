from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import setup, Extension

ext_modules = [
    Extension('faster_numpy.cython', sources=['faster_numpy/cython.pyx']),
    Extension('faster_numpy.clibrary', sources = ['packages/clibrary.cpp'])
]

setup(
    name='Speeding up numpy calculation',
    packages=['faster_numpy'],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
