from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name='BMI203-HW3',
  ext_modules=cythonize("hw3/_alignment.pyx"),
  include_dirs=[np.get_include()],
)
