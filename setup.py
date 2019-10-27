from distutils.core import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='rmbs',
      version='0.1.0',
      url='https://github.com/charleszxiong/rmbs',
      description='Reconstruct Many-Body States from Correlation Functions',
      long_description=long_description,
      author='Charles Zhaoxi Xiong',
      author_email='xiongzhaoxi@gmail.com',
      license='MIT',
      packages=['tensornet',
                'reconstruct'
                ],
      requires=['numpy',
                'scipy',
                'pandas'
                ],
      package_data={'tensornet': ['data/*'],
                    'reconstruct': ['data/*']
                    }
      )