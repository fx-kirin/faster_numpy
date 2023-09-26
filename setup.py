from setuptools import Extension, setup

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension('faster_numpy.cylib', sources=['faster_numpy/cylib.pyx'], include_dirs=[numpy.get_include()]),
    Extension('faster_numpy.clibrary', sources=['packages/clibrary.cpp'], include_dirs=[numpy.get_include()])
]

setup(
    name='faster_numpy',
    version="0.1.3",
    packages=['faster_numpy'],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext},
    package_data={
        '': ['*.pyx', 'pyproject.toml'],
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
