"""Atomic class handling."""

from __future__ import print_function

from . import sympify, KetBase
from ..multilevel.spin import JzKet

__all__ = [
    'Atom'
]

# TODO: Add multilevel support.
# See .multilevel

#-----------------------------------------------------------------------------
# Helper Classes and Functions
#-----------------------------------------------------------------------------

class Level():
    """Define's an atomic energy level.

    Does not define a complete symbolic level as in sympy.
    Only useful within the context of the Atom class."""

    def __init__(self, E, level_ket, label, atomic_label):
        # We're assuming that all error handling
        # has been done outside this class.
        self.E = E
        self.level_ket = level_ket
        self.label = label
        self.atomic_label = atomic_label

    def print(self, sep='\t'):
        out = '{0}' + sep + '{1}' + sep + '{2}' + sep + str({3})
        out = out.format(
            self.label,
            self.atomic_label,
            self.E,
            self.level_ket
        )
        
#-----------------------------------------------------------------------------
# Atom Class
#-----------------------------------------------------------------------------

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
    _labels = set()

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
    def labels(self):
        """Returns a copy of the unique level labels in the atom."""
        return self._labels.copy()

    @property
    def atomic_labels(self):
        """Returns a copy of the unique atomic labels in the atom."""
        return self._atomic_labels.copy()

    @property
    def state(self):
        """The atomic density matrix."""
        return self._state

    @property 
    def hamiltonian(self):
        """The hamiltonian for the atom."""
        return self._hamiltonian

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

    def __init__(self, **kwargs):
        # sympify the kwargs
        for key in kwargs:
            if key == 'name':
                kwargs[key] = str(kwargs[key])
            #TODO: error checking
            else:
                kwargs[key] = sympify(kwargs[key])

        self.kwargs = kwargs
        self._levels_list = [] # a list of the added levels
        self._levels = 0
        self._state = 0
        self._hamiltonian = 0

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

    def add_level(self, E, n, s, l, j, m_j, label=None):
        """Add an atomic energy level to the atom. Returns the atomic level
        ket.

        Parameters
        ==========

        E : Number, Symbol
            The atomic energy level.

        n : Int, Symbol
            The atomic energy level number, or principle number.

        s: Number, Symbol
            The total electron spin of the atomic level.

        l : Int, String, Symbol
            The atomic level's L total angular momentum.
            Accepts an integer or an atomic label ('S', 'P', etc).

        j : Number, Symbol
            The atomic level's J total angular momentum.

        m_j : Number, Symbol
            The atomic level's eigenstate of the Jz operator.

        label: String, Optional
            The unique label for the state. 
            This will be used in the notation of the density matrix.
            If none is given, the label will divert to the atomic label.

        Examples
        ========
        """
        # TODO: Make examples
        
        # TODO: Create automatic labels for 'E'
        E = sympify(E)
        n = sympify(n)
        s = sympify(s)
        #l = l
        j = sympify(j)
        m = sympify(m_j)

        # convert l to an atomic label and error check it
        l_labels = ['S', 'P', 'D', 'F']
        if isinstance(l, str):
            if str(l) in l_labels:
                l = sympify(l_labels.index(l))
            elif ord(l) <= ord('Z') and ord(l) > ord('F') and len(l) == 1:
                l = sympify(ord(l))
            else:
                raise ValueError('l should be a label, integer, or half integer, got: %s' % l)
        else:
            l = sympify(l)
        
        # make the level ket, also error checks
        level_ket = JzKet(n, s, l, j, m_j)

        # get the label necessary for the atomic label
        if l.is_Number:
            if l < 4:
                l_label = l_labels[l]
            elif l >= 4:
                l_label = chr(ord('F') + l - 4)
        
        # The atomic label
        # This differes from the user defined label,
        # which is used to make the notation of the density
        # matrix more convenient
        atomic_label = str(n) + l_label + '_' + str(j) + '(m_j=' + str(m_j) + ')'
        if atomic_label not in self._atomic_labels:
            self._atomic_labels.add(atomic_label)
        else:
            raise ValueError('The level you are adding already exists in this atom: %s' % atomic_label)

        # check if the label exists
        if label is not None:
            label = str(label)
        else:
            label = atomic_label

        # check its uniqueness
        if label not in self._labels:
            self._labels.add(label)
        else:
            raise ValueError(
                'The label you are adding already exists in this atom.\n'
                + 'Existing labels: %s, given: %s' % (self.labels, label)
                )

        # define the hamiltonian
        level_op = level_ket * level_ket.dual
        self._hamiltonian += E * level_op

        # add to the level
        new_level = Level(E, level_ket, label, atomic_label)
        self._levels_list.append(new_level)
        self._levels += 1
