"""Atomic class handling."""

from __future__ import print_function

from . import sympify, range, Abs, sqrt, zeros, S
from ..multilevel.spin import JzKet

__all__ = [
    'Atom'
]


#-----------------------------------------------------------------------------
# Helper Classes and Functions
#-----------------------------------------------------------------------------


class AtomicState():
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
    """Calculates how many values exist between F=Abs(J-I) to F=J+I"""
    f_min = Abs(J-I)
    f_max = J + I
    f_diff = F_max - F_min
    size = 2*F_diff + 1
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
        """The internal magnetic moment of the atom."""
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

    @classmethod
    def transition_strengths(cls, g_state, e_state, decouple='J'):
        """Calculate the transition strengths between two atomic states.
        This process iterates over the F numbers of a state, where 
        F = |J-I|, ..., J+I and J is the J quantum number of the state.

        Returns a Matrix of relative transition strengths from the ground state g
        to the excited state e. 

        Parameters
        ==========

        g_state : AtomicLevel
            The ground atomic state.

        e_state : AtomicLevel
            The excited atomic state.

        decouple : String, Optional
            The atomic transition to decompose to, either J' or 'L'.
        """
        
        decouple = str(decouple)
        if decouple not in ['L', 'J']:
            raise ValueError("decouple param is not 'L' or 'J', got: %s" % decouple)

        # make our notation simpler
        I = cls.spin

        # assume our states are atomic states
        assert isinstance(g_state, AtomicState)
        assert isinstance(e_state, AtomicState)

        g_ket = g_state.level_ket
        e_ket = e_state.level_ket
        Jg = g_ket.j
        Je = e_ket.j
        E_diff = e_state.E - g_state.E

        # label the gamma symbol
        label_g = g_state.label
        label_e = e_state.label
        gamma = 'gamma_{' + label_g + ',' + label_e + '}'
        gamma = sympify(gamma)

        # get the number of F values for the ground and excited state
        size_gf, gf_values = f_values(Jg, I)
        size_ef, ef_values = f_values(Je, I)

        # create a matrix that will store a table of 
        result = zeros(size_gf*size_ef, size_ef)


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

    def add_level(self, E, n, s, l, j, m_j, label=None):
        """Add an atomic energy level to the atom. 
        Returns an AtomicLevel class representing the state.

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
            If none is given, the label will divert to a numbered index.

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
        m_j = sympify(m_j)

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
            raise ValueError(
                'The level you are adding already exists in this atom: %s' % atomic_label
            )

        # make the string label
        if label is None:
            label = 0
            while str(label) not in self._labels:
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


        
