from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.1'
DESCRIPTION = 'MeerKAT Radio Telescope simulation data generator'
LONG_DESCRIPTION = 'MeerKAT Radio Telescope simulation data generator'

# Setting up
setup(
    name="MeerKATgen",
    version=VERSION,
    author="Peter Xiangyuan Ma (PetchMa)",
    author_email="<peterxy.ma@gmail.com>",
    description=DESCRIPTION,    
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'jax', 'setigen'],
    keywords=['python', 'simulation', 'radio telescope'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)