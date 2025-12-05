

## https://github.com/pypa/sampleproject/blob/main/setup.py

from setuptools import setup, find_packages
import pathlib
here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name = 'pycelp',
    version = '2021.06.02',
    description = 'Calculates coronal line polarization in the multi-line no coherence case',
    long_description=long_description,
    author = "Tom Schad",
    author_email = "tschad@nso.edu",
    packages=find_packages(),
    install_requires=['numpy','numba'],
)
