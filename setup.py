
from setuptools import setup



setup(
    name='Face Classification',
    version='1.1',
    description='A python module to build face image classification.',
    author='Sajjad Mahdavi',
    packages=[
        "data",
        "model",
        "train"
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'torch'

    ],
)
