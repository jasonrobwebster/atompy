"""Atomic class handling."""

from __future__ import print_function

#from .atombase import AtomBase
from . import sympify

class Atom():
    """Define a new atom.

    Parameters
    ==========

    name : String
        The name of the atom (Rb, H, Hydrogen, etc).

    spin : Number, Symbol
        The spin of the nucleus.
        Default is 0.

    mu : Number, Symbol
        The magnetic moment of the atom.
        Default is 0.

    mass : Number, Symbol
        The mass of the atom.

    isotope : Int, Symbol
        The isotope number of the atom.

    protons : Int, Symbol
        The number of protons within the atom.

    neutrons : Int, Symbol
        The number of neutrons within the atom.

    electrons : Int, Symbol
        The number of electrons within the atom.
    """

    #-------------------------------------------------------------------------
    # Variables
    #-------------------------------------------------------------------------

    _atomic_labels = set()

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def name(self):
        """The name (or symbol) of the atom."""
        return self.kwargs.get('name', '')

    @property
    def spin(self):
        """The internal spin of the nucleus."""
        return self.kwargs.get('spin', 0)

    @property
    def mu(self):
        """The magnetic moment of the atom."""
        return self.kwargs.get('mu', 0)

    @property
    def mass(self):
        """The mass of the atom."""
        return self.kwargs.get('mass')

    @property
    def isotope(self):
        """The nuclear isotope of the atom."""
        return self.kwargs.get('isotope')

    @property
    def protons(self):
        """The number of protons."""
        return self.kwargs.get('protons')

    @property
    def neutrons(self):
        """The number of neutrons."""
        return self.kwargs.get('neutrons')

    @property
    def electrons(self):
        """The number of electrons."""
        return self.kwargs.get('electrons')

    @property
    def levels(self):
        """The number of atomic levels."""
        return self._levels

    @property
    def state(self):
        """The atomic density matrix."""
        return self._state

    #-------------------------------------------------------------------------
    # Class Methods
    #-------------------------------------------------------------------------

    @classmethod
    def hartree(cls):
        """The Hartree method for deterimining the energy levels and states."""
        pass

    @classmethod
    def hartree_fock(cls):
        """The Hartree-Fock method for deterimining the energy levels and states."""
        pass

    #-------------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        # sympify the args & kwargs
        args = sympify(args)
        for key in kwargs:
            if key == 'name':
                kwargs[key] = str(kwargs[key])
            #TODO: error checking
            else:
                kwargs[key] = sympify(kwargs[key])

        self.args = args
        self.kwargs = kwargs
        self._levels = 0
        self._state = 0

    def __repr__(self):
        if self.name != '':
            # init the output
            out = []

            # append the isotope number
            if self.isotope is not None:
                out.append('{}^{' + str(self.isotope) + '}')

            # append the neutron number
            if self.neutrons is not None:
                out.append('{}_{' + str(self.neutrons) + '}')

            # append the name
            out.append(self.name)

            # append the charge
            if self.electrons is not None and self.protons is not None:
                charge = self.protons - self.electrons
                if charge > 0:
                    out.append('^{' + str(abs(charge)) + '+}')
                elif charge < 0:
                    out.append('^{' + str(abs(charge)) + '-}')

            # join everything
            out = ''.join(out)
            return out
        else:
            out = 'Atom({0}={1}, {2}={3}, {4}={5})'.format(
                'levels', self.levels,
                'spin', self.spin,
                'mu', self.mu
            )
            return out
    
    #-------------------------------------------------------------------------
    # Operations
    #-------------------------------------------------------------------------

    def add_level(self, **kwargs):
        """Add an atomic energy level to the atom.

        Parameters
        ==========

        E : Number, Symbol
            The atomic energy level.

        n : Int, Symbol
            The atomic energy level number, or principle number.

        j : Number, Symbol
            The atomic level's total angular momentum.

        m : Number, Symbol
            The atomic level's eigenstate of the Jz operator.

        mode : String (Optional)
            Whether the level represents a L, J, F, etc mode.
        
        Examples
        ========
        #TODO: make examples
        """

        if not len(kwargs) >= 4:
            raise ValueError('There should be at least 4 arguments.')
        if len(kwargs) > 5:
            raise ValueError('Too many arguments.')
        
        E = kwargs.pop('E')
        n = kwargs.pop('n')
        j = kwargs.pop('j')
        m = kwargs.pop('m')
        mode = kwargs.get('mode', 'J')