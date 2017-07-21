from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('utilities.cts.fast_cts', sources=['utilities/cts/fast_cts.pyx'])
]

setup(
    name='beating-montezuma',
    include_dirs=[np.get_include()],   
    ext_modules=cythonize(extensions)
)
