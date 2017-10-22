"""Atomic class handling."""

from __future__ import print_function

from . import sympify, JzKet, JzBra

__all__ = [
    'Atom'
]

# TODO: Add multilevel support.
# See .multilevel

class Level():
    """Define's an atomic level.

    Does not define a complete symbolic level as in sympy.

    Parameters
    ==========

    E : Number, Symbol
        Energy of the state.

    n : Int, Symbol
        Level of the state.

    s : Number, Symbol
        The electron spin of the level.

    l : Int, Symbol
        The AM of the level.

    j : Number, Symbol
        The total AM of the level.

    m : Number, Symbol
        The Jz eigenvalue of the state.
    """

    def __init__(self, **kwargs):
        

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

        m : Number, Symbol
            The atomic level's eigenstate of the Jz operator.
        
        Examples
        ========
        #TODO: make examples
        """

        if not len(kwargs) >= 6:
            raise ValueError('There should be at least 6 arguments, got: %s' % len(kwargs))
        
        E = sympify(kwargs.pop('E'))
        n = sympify(kwargs.pop('n'))
        s = sympify(kwargs.pop('s'))
        l = sympify(kwargs.pop('l'))
        j = sympify(kwargs.pop('j'))
        m = sympify(kwargs.pop('m'))

        # convert l to a atomic label and error check it
        l_labels = ['S', 'P', 'D', 'F']
        if str(l) in l_labels:
            l = sympify(l_labels.index(str(l)))
        elif ord(str(l)) <= ord('Z') and ord(str(l)) > ord('F'):
            l = sympify(ord(str(l)))
        else:
            if l.is_number:
                if (l) != int(l):
                    raise ValueError('l should be a label or integer got: %s' % l)
                if l in l_labels:
                    l_label = l_labels.index(l)
                else:
                    l_label = chr(ord('F') + l-4)
            else:
                raise ValueError('l should be a label, integer, or half integer, got: %s' % l)

        # label to add to atomic labels
        label = str(n) + l_label + '_' + str(j)
        if label not in self._atomic_labels:
            self._atomic_labels.add(label)
        else:
            raise ValueError('The level you are adding already exists in this atom: %s' % label)

        # define the hamiltonian
        level_op = JzKet(j, m) * JzBra(j, m)
        self._hamiltonian += E * level_op
        self._levels += 1