"""Atomic class handling."""

from __future__ import print_function

from sympy.core.compatibility import range
from sympy.matrices import zeros
from . import sympify, Abs, sqrt, S, clebsch_gordan, wigner_6j, m_values
from .tensor import SphericalTensor
from .doublebar import DoubleBar
from atompy.multilevel import AtomicJzKet


__all__ = [
    'AtomicState',
    'Atom'
]


#-----------------------------------------------------------------------------
# Helper Classes and Functions
#-----------------------------------------------------------------------------


class AtomicState():
    """Defines an atomic energy level.

    Parameters
    ==========

    E : Number, Symbol
        The energy eignestate of the atomic level.

    level_ket : AtomicJzKet, JzKet
        The atomic ket |n, s, l, j, m_j> that defines the level.

    label : String
        A label for this level.

    atomic_label : String, Optional
        The corresponding atomic label n^(2s+1)L_J.
    """

    def __init__(self, E, ket, label, atomic_label=None):
        # We're assuming that all error handling
        # has been done outside this class.
        self.E = E
        self.ket = ket
        self.label = label
        # TODO: bring the label and atomic_label functionality here.
        self.atomic_label = atomic_label

    def __repr__(self, sep='\t'):
        out = '{0}' + sep + '{1}' + sep + '{2}' + sep + str({3})
        out = out.format(
            self.label,
            self.atomic_label,
            self.E,
            self.level_ket
        )
        return out

def f_values(J, I):
    """Calculates how many F values exist between F=Abs(J-I) to F=J+I"""
    f_min = Abs(J-I)
    f_max = J + I
    f_diff = f_max - f_min
    size = 2*f_diff + 1
    if not size.is_Integer or size < 0:
        raise ValueError('size should be an integer, got J, I, size: %s, %s, %s' % (J, I, size))
    return size, [f for f in range(f_min, f_max + 1)]


#-----------------------------------------------------------------------------
# Atom Class
#-----------------------------------------------------------------------------


class Atom():
    """Define a new atom.

    Parameters
    ==========

    name : String
        The name of the atom (Rb, H, Hydrogen, etc).

    I : Number, Symbol
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
        return self.kwargs.get('spin', S.Zero)

    @property
    def mu(self):
        """The internal nuclear magnetic moment of the atom."""
        return self.kwargs.get('mu', S.Zero)

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

    def add_level(self, **kwargs):
        """Add an atomic energy level to the atom. 
        Returns an AtomicLevel class representing the state.
        All parameters are passed as keyword arguments.

        Parameters
        ==========

        energy : Number, Symbol
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
            If the spin of the atom is non zero, this becomes a coupling
            paramter for l and s.

        f : Number, Symbol
            The atomic level's F total angular momentum.
            This argument is not required if the atomic spin is zero.

        m : Number, Symbol
            The atomic level's eigenstate of the Jz operator. If the atomic
            spin is non zero, this is the eigenstate of the Fz operator.

        label: String, Optional
            The unique label for the state.
            This will be used in the notation of the density matrix.
            If none is given, the label will divert to a numbered index.

        Examples
        ========
        """
        # TODO: Make examples

        # TODO: Create automatic labels for 'E'
        E = sympify(kwargs['energy'])
        n = sympify(kwargs['n'])
        s = sympify(kwargs['s'])
        l = kwargs['l']
        j = sympify(kwargs['j'])
        m = sympify(kwargs['m'])
        label = kwargs.get('label')
        if self.spin != 0:
            f = sympify(kwargs['f'])

        if E.is_Number:
            if not E.is_real:
                raise ValueError('The energy of the level must be real, got %s' % E)

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

        # make the level ket, also  performs error checking
        i = self.spin
        if i == 0:
            level_ket = AtomicJzKet(n, j, m, (s, l))
        else:
            if f.is_number and j.is_number and i.is_number:
                if abs(j-i) > f or abs(j+i) < f:
                    raise ValueError(
                        'f must be in the bounds |j-i|<=f<=|j+i|, got j, i, f: %s, %s, %s' % (j, i, f)
                    )
            level_ket = AtomicJzKet(n, f, m, (s, l, i), [(1, 2, j), (1, 3, f)])

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
        if i == 0:
            atomic_label = str(n) + l_label + '_' + str(j) + '(m_j=' + str(m) + ')'
        else:
            atomic_label = str(n) + l_label + '_' + str(j) + '(F=' + str(f) + ',m_f=' + str(m) + ')'
        if atomic_label not in self._atomic_labels:
            self._atomic_labels.add(atomic_label)
        else:
            raise ValueError(
                'The level you are adding already exists in this atom: %s' % atomic_label
            )

        # make the string label
        if label is None:
            label = 0
            while str(label) in self._labels:
                label += 1
        label = str(label)

        # check its uniqueness
        if label not in self._labels:
            self._labels.add(label)
        else:
            raise ValueError(
                'The given label already exists in this atom.\n'
                + 'Existing labels: %s, given: %s' % (self.labels, label)
                )

        # define the hamiltonian
        level_op = level_ket * level_ket.dual
        self._hamiltonian += E * level_op

        # add to the level
        new_level = AtomicState(E, level_ket, label, atomic_label)
        self._levels_list.append(new_level)
        self._levels += 1

        return new_level
    
    # TODO: Add a method to calculate zeeman splitting
    # TODO: Integrate support for adding all m_j sublevels for a given j
    # TODO: Add method to calculate spin orbit coupling
    # TODO: Add support for adding all j and m_j sublevels for a given L and S
    # TODO: Add a method to calculate hyperfine splitting
