from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("PPOAgent", ["PPOAgent.pyx"],
        include_dirs=["./"],
        #libraries=["tensorflow_framework"], 
        library_dirs=["./"]),
]

setup(
    name='PPO Agent API',
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
