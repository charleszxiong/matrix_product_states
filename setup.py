from distutils.core import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name='mps',
    version='0.1.0',
    url='https://github.com/charleszxiong/mps',
    description='Reconstruct Quantum Many-Body States from Correlation Functions',
    long_description=long_description,
    author='Charles Zhaoxi Xiong',
    author_email='xiongzhaoxi@gmail.com',
    license='MIT',
    packages=['mps', 'mps.models'],
    requires=[
        'numpy >= 1.16.3',
        'scipy >= 1.3.0',
        'pandas >= 0.24.2'
    ]
)
