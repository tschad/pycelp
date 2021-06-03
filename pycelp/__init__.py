"""
pycelp is a python package for Coronal Emission Line Polarization calculations.
It is used to forward synthesize the polarized emission of ionized atoms
formed in the solar corona. It calculates the atomic density matrix elements
for a single ion under coronal equilibrium conditions and excited by a
prescribed radiation field and thermal collisions. In its initial release,
pyCELP solves a set of statistical equilibrium equations in the spherical
statistical tensor respresentation for a multi-level atom for the no-coherence
case. This approximation is useful in the case of forbidden line emission by
visible and infrared lines, such as Fe XIII 1074.7 nm and Si X 3.9 um.
See Schad & Dima 2020 for more details and specific references.

.. include:: ./package_documentation.md
"""

from .ion import Ion
