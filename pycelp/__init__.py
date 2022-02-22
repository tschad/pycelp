"""
.. include:: ../README.md
"""
from .ion import Ion

import numpy as np 

class vanVleck_angles:
    def __init__(self):  
        self.rad = np.arccos(1./np.sqrt(3.))
        self.deg = np.rad2deg(self.rad)
    
vanVleck = vanVleck_angles()

__all__ = ['Ion']
