
## Preface

[![github](https://img.shields.io/badge/GitHub-tschad%2FpyCELP-blue.svg?style=flat)](https://github.com/tschad/pycelp)
[![ADS](https://img.shields.io/badge/NASA%20ADS-SoPh%2C%20V295%2C%207%2C%2098-red)](https://ui.adsabs.harvard.edu/abs/2020SoPh..295...98S/abstract)

A **py**thon package for **C**oronal **E**mission **L**ine **P**olarization calculations.

Lead Developer: T. Schad - National Solar Observatory

**DISCLAIMER: pycelp is still in the early stages of development. Contributors are welcome. **

## About the code

pyCELP is used to forward synthesize the polarized emission of ionized atoms formed in the solar corona.  It calculates the atomic density matrix elements for a single ion under coronal equilibrium conditions and excited by a prescribed radiation field and thermal collisions.  In its initial release, pyCELP solves a set of statistical equilibrium equations in the spherical statistical tensor respresentation for a multi-level atom for the no-coherence case.  This approximation is useful in the case of forbidden line emission by visible and infrared lines, such as Fe XIII 1074.7 nm and Si X 3.9 um.   See
[Schad & Dima 2020](https://ui.adsabs.harvard.edu/abs/2020SoPh..295...98S/abstract) for more details and specific references.

A read-only Enhanced PDF version of Schad & Dima 2020 is available via this [link](https://rdcu.be/b5J2X).

The original code developed by [Schad & Dima 2020](https://ui.adsabs.harvard.edu/abs/2020SoPh..295...98S/abstract) (previously referred to as pyCLE) was a Fortran based code wrapped in python.  pyCELP is a completely new implementation coded entirely in Python.  It takes advantage of specific algorithm changes, numba jit compilers, and efficient numpy linear algebra packages to provide excellent speed performance that in most cases exceeds the earlier code.  More information pertaining to numba is below.

## Install

### Dependencies

* python3, numpy, numba
* (optional - for tests/examples) matplotlib, scipy
* (optional - for updating docs) pdoc3
* The [CHIANTI atomic database](http://www.chiantidatabase.org/chianti_download.html) is also required.  (Currelty tested with v9) pyCELP will automatically search for the Chianti atomic database path using the default environment variable XUVTOP.


### Conda environment

It is recommended to install pycelp within a conda environment.  For the best performance, it is recommended to use a version of numpy with an optimal linear algebra library, e.g. MKL for intel compilers (https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries).

Example:
```shell
$ conda create --name pycelp
$ conda activate pycelp
$ conda install python numpy scipy numba matplotlib
```

### Download/clone repo

```shell
$ git clone https://github.com/tschad/pycelp.git
$ cd pycelp
$ python setup.py develop  
```

## Examples

Below is a minimal example of using the pycelp code from a python terminal.  For
more extensive examples, see those provided in the examples subdirectory
within the project repo.

```shell
(juplab) [schad@Schad-Mac pycelp]$ python
Python 3.9.4 (default, Apr  9 2021, 09:32:38)
[Clang 10.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pycelp
>>> fe13 = pycelp.Ion('fe_13',nlevels = 50)
 reading:  /usr/local/ssw/packages/chianti/dbase/fe/fe_13/fe_13.elvlc
 reading:  /usr/local/ssw/packages/chianti/dbase/fe/fe_13/fe_13.wgfa
 reading:  /usr/local/ssw/packages/chianti/dbase/fe/fe_13/fe_13.scups
 reading:  /usr/local/ssw/packages/chianti/dbase/fe/fe_13/fe_13.psplups
 using default abundances: /usr/local/ssw/packages/chianti/dbase/abundance/sun_photospheric_2009_asplund.abund
 reading:  /usr/local/ssw/packages/chianti/dbase/abundance/sun_photospheric_2009_asplund.abund
 testing default file: /usr/local/ssw/packages/chianti/dbase/ioneq/chianti.ioneq
 reading:  /usr/local/ssw/packages/chianti/dbase/ioneq/chianti.ioneq
 setting up electron collision rate factors
 setting up proton  collision rate factors
 setting up non-dipole radiative rate factors
 getting non-dipole rate factors
 setting up dipole radiative rate factors
>>>
>>> fe13
pyCELP Ion class
    ---------------------
    Ion Name: fe_13
    Number of energy levels included: 50
    Number of SEE equations: 142
    Number of Radiative Transitions: 366
    Ionization Equilbrium Filename: /usr/local/ssw/packages/chianti/dbase/ioneq/chianti.ioneq
>>>
```

## Numba implementation and options

pyCELP uses numba @njit decorators for jit compiling many portions of the codebase.  In most instances, the code adopts a file-based cache for storing compiled versions of the code for later use.  The first time pyCELP is used, there is additional overhead in the time required to compile the code.  Subsequent calls are significantly faster.  If one makes modifications to the code and errors occur, it may be advised to delete the cached files the \__pycache__ directory of the installed package.

The code does not use numba parallel options for multithreading.

Numba can be disabled through the use of an environmental variable (NUMBA_DISABLE) but this is not frequently used.

## Numpy libraries for multiprocessing

pyCELP uses numpy libraries which can have multithreaded modules.  If pyCELP is used in a multiprocessor application, threads need to be properly managed.

## Updating documentation

Code reference documentation is available at [tschad.github.io/pycle](https:://tschad.github.io/pycle).  These are created using pdoc3.  They are easily manually built and/or updated from the main project repo directory by using the following command.  

```shell
pdoc --html --force --output-dir docs pycelp
```

## Acknowledgements

pyCELP evolved from work initially using the CLE code developed by Phil Judge
and Roberto Casini at the High Altitude Observatory.  While pyCELP is now a
completely independent implementation, we express our gratitude for all we
learned by using CLE.  pyCELP has been developed based on the excellent
treatise on spectral line polarization by Egidio Landi Degl’innocenti and Marco
Landolfi available [here](https://link.springer.com/book/10.1007/1-4020-2415-0).
