"""Init"""

# import sympy at the core level
# TODO: rework atompys level structure
from sympy import sympify, symbols, Eq, I, Function, Symbol
from sympy.core import Add, Mul, S
from sympy.printing import *
from sympy.concrete import *
from sympy.functions import *
from sympy.physics.wigner import *
from sympy.physics.quantum import *
from sympy.physics.quantum.cg import *
from sympy.physics.quantum.spin import *
from sympy.physics.quantum.hilbert import *
from sympy.physics.units import (
    hbar, speed_of_light as c, e0, u0, eV,
    joule, meter, second, kilogram
)

#NIST CODATA values 2017
#eV = Quantity("eV", joule.dimension, 1.6021766208e-19*joule) 
#c = Quantity("c", (meter/second).dimension, 299792458 * meter/second)
#e0 = Quantity("epsilon_0", (farad/meter).dimension, 8.854187817e-12 * farad/meter)
#u0 = Quantity("mu_0", (meter*kilogram/(second**2)/(ampere**2)).dimension, 4*pi*1e-7 * meter*kilogram/(second**2)/(ampere**2))

# import self
from .atomicstate import *
from .doublebar import *
from .tensor import *
