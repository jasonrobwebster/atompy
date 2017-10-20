"""
Atompy is a python package used in atomic physics based around the
SymPy package. 
"""

from __future__ import print_function

# import sympy
from sympy import init_printing
from sympy.functions import *
from sympy.physics.quantum import *
from sympy.physics.quantum.cg import *
from sympy.physics.quantum.spin import *
from sympy.physics.quantum.hilbert import *

# import self
from .core import *

init_printing()
