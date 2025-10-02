from setuptools import setup, find_packages

setup(
    name='is3',
    version='0.0.1',
    author='cberger',
    install_requires=['torchaudio>=2.1.2',
                      'torch>=2.1.2',
                      'lightning>=2.2.1',
                      'pandas',
                      ],
    packages=find_packages(include=['is3*'])
)
