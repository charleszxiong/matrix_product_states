from distutils.core import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name='rmbs',
    version='0.1.0',
    url='https://github.com/charleszxiong/rmbs',
    description='Reconstruct Quantum Many-Body States from Correlation Functions',
    long_description=long_description,
    author='Charles Zhaoxi Xiong',
    author_email='xiongzhaoxi@gmail.com',
    license='MIT',
    packages=['rmbs', 'rmbs.models'],
    requires=[
        'numpy >= 1.16.3',
        'scipy >= 1.3.0',
        'pandas >= 0.24.2'
    ]
)